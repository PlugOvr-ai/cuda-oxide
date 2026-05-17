/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WGMMA (Warpgroup Matrix Multiply-Accumulate) intrinsic conversion for Hopper+ GPUs.
//!
//! # Operations
//!
//! | Operation             | PTX                             | Description                    |
//! |-----------------------|---------------------------------|--------------------------------|
//! | `Fence`               | `wgmma.fence.sync.aligned`      | Memory fence before WGMMA      |
//! | `CommitGroup`         | `wgmma.commit_group.sync.aligned`| Commit pending operations     |
//! | `WaitGroup`           | `wgmma.wait_group.sync.aligned N`| Wait for N groups             |
//! | `MakeSmemDesc`        | cvta + bit manipulation         | Create shared memory descriptor|
//! | `MmaM64N64K16F32Bf16` | `wgmma.mma_async`               | Matrix multiply                |

use crate::convert::intrinsics::common::*;
use dialect_llvm::ops as llvm;
use dialect_llvm::ops::GepIndex;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::value::Value;

pub(crate) fn convert_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "wgmma.fence.sync.aligned;",
        "",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_commit_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "wgmma.commit_group.sync.aligned;",
        "",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert WGMMA wait_group to inline PTX.
pub(crate) fn convert_wait_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("wgmma_wait_group requires 1 operand");
    }
    let n = operands[0];

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![n],
        "wgmma.wait_group.sync.aligned $0;",
        "n",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert WGMMA make_smem_desc to inline PTX.
pub(crate) fn convert_make_smem_desc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("wgmma_make_smem_desc requires operand");
    }
    let ptr = operands[0];
    let ptr_casted = cast_to_shared_addrspace(ctx, rewriter, ptr);

    let asm_template = r#"{
    .reg .u64 addr;
    cvta.to.shared.u64 addr, $1;
    shr.u64 addr, addr, 4;
    and.b64 addr, addr, 0x3FFF;
    or.b64 $0, addr, 0xC000000800080000;
}"#;

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i64_ty.into(),
        vec![ptr_casted],
        asm_template,
        "=l,l",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert WGMMA MMA operation to inline PTX.
///
/// The full lowering requires register allocation for 16+ output registers
/// and is not yet implemented. Until it lands, calls to
/// `cuda_device::wgmma::wgmma_mma_*` from a `#[kernel]` are rejected at
/// codegen time with a clear diagnostic.
///
/// The previous behaviour silently emitted `// wgmma.mma placeholder` as an
/// inline-asm comment and erased the op, producing PTX that loaded and ran
/// but multiplied-accumulated to zero — a silent miscompile with no warning.
pub(crate) fn convert_mma(
    _ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    _op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    pliron::input_err_noloc!(
        "wgmma.mma_async lowering is not yet implemented; \
         calls to `cuda_device::wgmma::wgmma_mma_*` from a kernel are \
         currently unsupported. Tracking issue: full lowering requires \
         register allocation for 16+ output registers."
    )
}

/// Convert Ampere `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`.
///
/// Operands: `[acc_ptr, a0, a1, a2, a3, b0, b1]`. Loads the 4-f32
/// accumulator from `acc_ptr`, issues the tensor-core MMA with the C/D
/// accumulator tied (read-modify-write), and stores the 4 results back.
pub(crate) fn convert_mma_sync(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<Value> = op.deref(ctx).operands().collect();
    if operands.len() != 7 {
        return pliron::input_err_noloc!(
            "mma_sync_m16n8k8_f32_tf32 requires 7 operands [acc_ptr, a0..a3, b0, b1]"
        );
    }
    let acc_ptr = operands[0];
    let (a0, a1, a2, a3) = (operands[1], operands[2], operands[3], operands[4]);
    let (b0, b1) = (operands[5], operands[6]);

    let f32_ty = FP32Type::get(ctx);

    // Load c0..c3 from the accumulator (acc_ptr is f32*; index by element).
    let mut c = Vec::with_capacity(4);
    for i in 0..4u32 {
        let gep = llvm::GetElementPtrOp::new(
            ctx,
            acc_ptr,
            vec![GepIndex::Constant(i)],
            f32_ty.into(),
        )?;
        rewriter.insert_operation(ctx, gep.get_operation());
        let gptr = gep.get_operation().deref(ctx).get_result(0);
        let ld = llvm::LoadOp::new(ctx, gptr, f32_ty.into());
        rewriter.insert_operation(ctx, ld.get_operation());
        c.push(ld.get_operation().deref(ctx).get_result(0));
    }

    // D = A·B + C. C/D are the 4 tied accumulator registers ($0..$3);
    // A = {$4,$5,$6,$7}, B = {$8,$9}.
    let asm = llvm::InlineAsmMultiOp::new_tied_convergent(
        ctx,
        4,
        f32_ty.into(),
        c.clone(),
        vec![a0, a1, a2, a3, b0, b1],
        // Operand numbering: $0..$3 = outputs (D). Inputs follow: the 4 tied
        // accumulator inputs are $4..$7 (alias $0..$3), then A = $8..$11,
        // B = $12,$13. C uses the tied outputs $0..$3.
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \
{$0,$1,$2,$3}, {$8,$9,$10,$11}, {$12,$13}, {$0,$1,$2,$3};",
        "f",
        "r,r,r,r,r,r",
    );
    rewriter.insert_operation(ctx, asm.get_operation());

    // Store the 4 D results back into the accumulator.
    for i in 0..4usize {
        let d = asm.get_operation().deref(ctx).get_result(i);
        let gep = llvm::GetElementPtrOp::new(
            ctx,
            acc_ptr,
            vec![GepIndex::Constant(i as u32)],
            f32_ty.into(),
        )?;
        rewriter.insert_operation(ctx, gep.get_operation());
        let gptr = gep.get_operation().deref(ctx).get_result(0);
        let st = llvm::StoreOp::new(ctx, d, gptr);
        rewriter.insert_operation(ctx, st.get_operation());
    }

    rewriter.erase_operation(ctx, op);
    Ok(())
}
