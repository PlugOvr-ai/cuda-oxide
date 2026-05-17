/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WGMMA conversion for Hopper `sm_90a`.

use crate::convert::intrinsics::common::*;
use dialect_llvm::ops as llvm;
use dialect_llvm::ops::GepIndex;
use llvm_export::types::VoidType;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::value::Value;

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

fn accumulator_register_list() -> String {
    (0..32)
        .map(|index| format!("%acc{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn deferred_group_template(mma_count: usize) -> String {
    let mut template = String::from("{\n    .reg .f32 %acc<32>;\n");

    for index in 0..32 {
        let offset = index * 4;
        template.push_str(&format!("    ld.f32 %acc{index}, [$0 + {offset}];\n"));
    }

    template.push_str("    wgmma.fence.sync.aligned;\n");
    let registers = accumulator_register_list();
    for mma_index in 0..mma_count {
        let desc_a = 1 + mma_index * 2;
        let desc_b = desc_a + 1;
        template.push_str(&format!(
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
             {{{registers}}}, ${desc_a}, ${desc_b}, 1, 1, 1, 0, 0;\n"
        ));
    }
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");

    for index in 0..32 {
        let offset = index * 4;
        template.push_str(&format!("    st.f32 [$0 + {offset}], %acc{index};\n"));
    }
    template.push('}');
    template
}

/// Lower a complete deferred BF16 WGMMA group.
///
/// The inline-PTX scope owns 32 explicit accumulator registers. It loads them
/// before the fence, issues every MMA, commits, waits for zero pending groups,
/// and writes them back only after the wait. This avoids exposing pending
/// accumulator values to LLVM or to memory.
pub(crate) fn convert_mma_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 || operands.len() % 2 == 0 {
        return pliron::input_err_noloc!(
            "deferred WGMMA group requires one accumulator pointer and one or more descriptor pairs"
        );
    }

    let mma_count = (operands.len() - 1) / 2;
    let template = deferred_group_template(mma_count);
    let mut constraints = vec!["l"; operands.len()];
    constraints.push("~{memory}");
    let constraints = constraints.join(",");

    inline_asm_convergent(
        ctx,
        rewriter,
        VoidType::get(ctx).into(),
        operands,
        &template,
        &constraints,
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Reject an unfused pointer-form MMA operation.
///
/// Reaching this converter means the pre-lowering adapter could not prove a
/// complete and sound fence/MMA/commit/wait sequence.
pub(crate) fn convert_mma(
    _ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    _op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    pliron::input_err_noloc!(
        "WGMMA MMA reached lowering without deferred accumulator fusion; expected a linear fence -> BF16 MMA+ -> commit_group -> wait_group<0> sequence"
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

#[cfg(test)]
mod tests {
    use super::deferred_group_template;

    #[test]
    fn deferred_template_keeps_loads_before_wait_and_stores_after_wait() {
        let template = deferred_group_template(2);
        assert_eq!(template.matches("ld.f32 %acc").count(), 32);
        assert_eq!(template.matches("st.f32 [$0").count(), 32);
        assert_eq!(template.matches("wgmma.mma_async").count(), 2);

        let first_mma = template.find("wgmma.mma_async").unwrap();
        let wait = template.find("wgmma.wait_group.sync.aligned 0").unwrap();
        let first_store = template.find("st.f32 [$0").unwrap();
        assert!(first_mma < wait);
        assert!(wait < first_store);
    }
}
