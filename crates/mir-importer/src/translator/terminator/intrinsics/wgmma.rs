/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate) intrinsics.
//!
//! Handles Hopper `sm_90a` asynchronous warpgroup matrix operations.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    MmaSyncM16N8K8F32Tf32Op, MmaSyncM16N8K16F32F16Op, MmaSyncM16N8K32S32S8Op,
    WgmmaMakeSmemDescOp,
    WgmmaMmaM64N64K16F32Bf16Op, LdmatrixX4B16Op,
    LdmatrixX1B16Op, LdmatrixX2B16Op, LdmatrixX2TransB16Op, LdmatrixX4TransB16Op, CpAsyncCg16Op, CpAsyncCommitGroupOp, CpAsyncWaitAllOp, MovmatrixTransB16Op,};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

const CUSTOM_DESCRIPTOR_UNSUPPORTED: &str = "custom WGMMA descriptor encoding is not yet supported";
const MMA_UNSUPPORTED: &str = "this WGMMA MMA variant is not yet supported; only m64n64k16.f32.bf16.bf16 has deferred accumulator lowering";

fn unsupported_diagnostic(path: &str) -> Option<&'static str> {
    match path {
        "cuda_device::wgmma::make_smem_desc_custom" => Some(CUSTOM_DESCRIPTOR_UNSUPPORTED),
        "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_f16"
        | "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_tf32" => Some(MMA_UNSUPPORTED),
        _ => None,
    }
}

/// Reject public WGMMA entries that do not have a sound lowering yet.
pub(crate) fn reject_unsupported(path: &str, loc: Location) -> TranslationResult<()> {
    let Some(diagnostic) = unsupported_diagnostic(path) else {
        return Ok(());
    };
    input_err!(loc, TranslationErr::unsupported(diagnostic))
}

/// Emit make_smem_desc: Create SMEM descriptor for WGMMA.
pub fn emit_wgmma_make_smem_desc(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "make_smem_desc expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let desc_op = Operation::new(
        ctx,
        WgmmaMakeSmemDescOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![ptr_val],
        vec![],
        0,
    );
    desc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        desc_op.insert_after(ctx, prev);
    } else {
        desc_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = desc_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        desc_op,
        value_map,
        block_map,
        loc,
        "make_smem_desc call without target block",
    )
}

/// Emit BF16 m64n64k16 WGMMA pointer form.
///
/// `mir-lower` later fuses this operation with the surrounding fence, commit,
/// and `wait_group<0>` so the accumulator remains in registers until the wait.
pub fn emit_wgmma_mma_m64n64k16_f32_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "wgmma_mma_m64n64k16_f32_bf16 expects 3 arguments (acc_ptr, desc_a, desc_b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let (acc_ptr, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;
    let (desc_a, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;
    let (desc_b, next) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next;

    let mma_op = Operation::new(
        ctx,
        WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![acc_ptr, desc_a, desc_b],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, mma_op, block_map, loc))
    } else {
        input_err!(
            loc,
            TranslationErr::unsupported(
                "wgmma_mma_m64n64k16_f32_bf16 call without target block".to_string()
            )
        )
    }
}

/// Emit `mma_sync_m16n8k8_f32_tf32`: Ampere warp MMA (D = A·B + C).
///
/// Args: `[acc_ptr, a0, a1, a2, a3, b0, b1]`.  Returns: void (acc in place).
#[allow(clippy::too_many_arguments)]
pub fn emit_mma_sync_m16n8k8_f32_tf32(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 7 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_sync_m16n8k8_f32_tf32 expects 7 arguments \
                 (acc_ptr, a0..a3, b0, b1), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(7);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let mma_op = Operation::new(
        ctx,
        MmaSyncM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![], // No results (accumulator updated in-place via acc_ptr)
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "mma_sync_m16n8k8_f32_tf32 call without target block".to_string()
            )
        )
    }
}

/// Emit `mma_sync_m16n8k16_f32_f16`: Ampere warp MMA, f16 inputs (D = A·B + C).
///
/// Args: `[acc_ptr, a0, a1, a2, a3, b0, b1]`.  Returns: void (acc in place).
#[allow(clippy::too_many_arguments)]
pub fn emit_mma_sync_m16n8k16_f32_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 7 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_sync_m16n8k16_f32_f16 expects 7 arguments \
                 (acc_ptr, a0..a3, b0, b1), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(7);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let mma_op = Operation::new(
        ctx,
        MmaSyncM16N8K16F32F16Op::get_concrete_op_info(),
        vec![], // No results (accumulator updated in-place via acc_ptr)
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "mma_sync_m16n8k16_f32_f16 call without target block".to_string()
            )
        )
    }
}

/// Emit `mma_sync_m16n8k32_s32_s8`: Ampere warp MMA, s8 inputs, s32 accumulator.
///
/// Args: `[acc_ptr, a0, a1, a2, a3, b0, b1]`.  Returns: void (acc in place).
#[allow(clippy::too_many_arguments)]
pub fn emit_mma_sync_m16n8k32_s32_s8(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 7 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_sync_m16n8k32_s32_s8 expects 7 arguments \
                 (acc_ptr, a0..a3, b0, b1), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(7);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let mma_op = Operation::new(
        ctx,
        MmaSyncM16N8K32S32S8Op::get_concrete_op_info(),
        vec![], // No results (accumulator updated in-place via acc_ptr)
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "mma_sync_m16n8k32_s32_s8 call without target block".to_string()
            )
        )
    }
}

/// Emit `ldmatrix_x4_b16`: cooperative warp fragment load from shared memory.
pub fn emit_ldmatrix_x4_b16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ldmatrix_x4_b16 expects 2 arguments (out_ptr, addr), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let ld_op = Operation::new(
        ctx,
        LdmatrixX4B16Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, ld_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("ldmatrix_x4_b16 call without target block".to_string())
        )
    }
}

/// Emit `ldmatrix_x1_b16`.
pub fn emit_ldmatrix_x1_b16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ldmatrix_x1_b16 expects 2 arguments (out_ptr, addr), got {}",
                args.len()
            ))
        );
    }
    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }
    let new_op = Operation::new(ctx, LdmatrixX1B16Op::get_concrete_op_info(), vec![], operands, vec![], 0);
    new_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, new_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("ldmatrix_x1_b16 call without target block".to_string())
        )
    }
}

/// Emit `ldmatrix_x2_b16`.
pub fn emit_ldmatrix_x2_b16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ldmatrix_x2_b16 expects 2 arguments (out_ptr, addr), got {}",
                args.len()
            ))
        );
    }
    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }
    let new_op = Operation::new(ctx, LdmatrixX2B16Op::get_concrete_op_info(), vec![], operands, vec![], 0);
    new_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, new_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("ldmatrix_x2_b16 call without target block".to_string())
        )
    }
}

/// Emit `ldmatrix_x2_trans_b16`.
pub fn emit_ldmatrix_x2_trans_b16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ldmatrix_x2_trans_b16 expects 2 arguments (out_ptr, addr), got {}",
                args.len()
            ))
        );
    }
    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }
    let new_op = Operation::new(ctx, LdmatrixX2TransB16Op::get_concrete_op_info(), vec![], operands, vec![], 0);
    new_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, new_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("ldmatrix_x2_trans_b16 call without target block".to_string())
        )
    }
}

/// Emit `ldmatrix_x4_trans_b16`.
pub fn emit_ldmatrix_x4_trans_b16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ldmatrix_x4_trans_b16 expects 2 arguments (out_ptr, addr), got {}",
                args.len()
            ))
        );
    }
    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }
    let new_op = Operation::new(ctx, LdmatrixX4TransB16Op::get_concrete_op_info(), vec![], operands, vec![], 0);
    new_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, new_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("ldmatrix_x4_trans_b16 call without target block".to_string())
        )
    }
}

/// Emit `cp_async_shared_global_16`.
pub fn emit_cp_async_shared_global_16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_shared_global_16 expects 2 arguments (dst, src), got {}",
                args.len()
            ))
        );
    }
    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }
    let new_op = Operation::new(ctx, CpAsyncCg16Op::get_concrete_op_info(), vec![], operands, vec![], 0);
    new_op.deref_mut(ctx).set_loc(loc.clone());
    // A generated intrinsic: the target-requirements verifier rejects it
    // without its append-only ABI marker from intrinsics/abi-v1.toml
    // (cp.async.cg.shared.global, 16 bytes).
    super::super::helpers::set_generated_intrinsic_marker(ctx, new_op, "v1:i0092");
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, new_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("cp_async_shared_global_16 call without target block".to_string())
        )
    }
}

/// Emit `cp_async_commit_group`.
pub fn emit_cp_async_commit_group(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 0 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_commit_group expects 0 arguments (none), got {}",
                args.len()
            ))
        );
    }
    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(0);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }
    let new_op = Operation::new(ctx, CpAsyncCommitGroupOp::get_concrete_op_info(), vec![], operands, vec![], 0);
    new_op.deref_mut(ctx).set_loc(loc.clone());
    // A generated intrinsic: the target-requirements verifier rejects it
    // without its append-only ABI marker from intrinsics/abi-v1.toml
    // (cp.async.commit_group).
    super::super::helpers::set_generated_intrinsic_marker(ctx, new_op, "v1:i0094");
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, new_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("cp_async_commit_group call without target block".to_string())
        )
    }
}

/// Emit `cp_async_wait_all`.
pub fn emit_cp_async_wait_all(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 0 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_wait_all expects 0 arguments (none), got {}",
                args.len()
            ))
        );
    }
    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(0);
    for arg in args {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }
    let new_op = Operation::new(ctx, CpAsyncWaitAllOp::get_concrete_op_info(), vec![], operands, vec![], 0);
    new_op.deref_mut(ctx).set_loc(loc.clone());
    // A generated intrinsic: the target-requirements verifier rejects it
    // without its append-only ABI marker from intrinsics/abi-v1.toml
    // (cp.async.wait_all).
    super::super::helpers::set_generated_intrinsic_marker(ctx, new_op, "v1:i0095");
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, new_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("cp_async_wait_all call without target block".to_string())
        )
    }
}

/// Emit `movmatrix_trans_b16`: register-resident 8x8 b16 transpose.
pub fn emit_movmatrix_trans_b16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "movmatrix_trans_b16 expects 1 argument (value), got {}",
                args.len()
            ))
        );
    }
    let (val, last_op) =
        rvalue::translate_operand(ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone())?;
    // The device fn returns `u32`, so the result type must be *unsigned*;
    // a signless i32 fails the store's type check against the destination.
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let new_op = Operation::new(
        ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![val],
        vec![],
        0,
    );
    new_op.deref_mut(ctx).set_loc(loc.clone());
    // Generated intrinsic; see intrinsics/abi-v1.toml (movmatrix.m8n8.trans.b16).
    super::super::helpers::set_generated_intrinsic_marker(ctx, new_op, "v1:i0305");
    if let Some(prev) = last_op {
        new_op.insert_after(ctx, prev);
    } else {
        new_op.insert_at_front(block_ptr, ctx);
    }
    let result = new_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        new_op,
        value_map,
        block_map,
        loc,
        "movmatrix_trans_b16 call without target block",
    )
}

#[cfg(test)]
mod tests {
    use super::{CUSTOM_DESCRIPTOR_UNSUPPORTED, MMA_UNSUPPORTED, unsupported_diagnostic};

    #[test]
    fn unsupported_wgmma_paths_are_exact() {
        assert_eq!(
            unsupported_diagnostic("cuda_device::wgmma::make_smem_desc_custom"),
            Some(CUSTOM_DESCRIPTOR_UNSUPPORTED)
        );
        assert_eq!(
            unsupported_diagnostic("cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16"),
            None
        );
        for path in [
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_f16",
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_tf32",
        ] {
            assert_eq!(unsupported_diagnostic(path), Some(MMA_UNSUPPORTED));
        }

        for path in [
            "cuda_device::wgmma::make_smem_desc",
            "cuda_device::wgmma::wgmma_fence",
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16_extra",
            "other_crate::wgmma::wgmma_mma_m64n64k16_f32_bf16",
        ] {
            assert_eq!(unsupported_diagnostic(path), None);
        }
    }
}
