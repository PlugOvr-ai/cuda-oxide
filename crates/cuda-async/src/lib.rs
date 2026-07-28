/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Async execution layer for CUDA device operations.
//!
//! This crate provides a futures-based interface for composing, scheduling, and
//! executing GPU work. The core abstraction is [`DeviceOperation`], a lazy,
//! composable representation of GPU work that is decoupled from any particular
//! CUDA stream or device until the moment it is scheduled.
//!
//! # Architecture
//!
//! ```text
//!   DeviceOperation  ──schedule()──>  DeviceFuture  ──.await──>  Result<T>
//!         │                                │
//!    (lazy graph)                    (stream assigned)
//!         │                                │
//!    combinators:                    SchedulingPolicy
//!    and_then, zip,                  picks the stream
//!    with_context, ...
//! ```
//!
//! ## DeviceOperation model
//!
//! A [`DeviceOperation`] is a lazy description of GPU work. It carries no stream
//! affinity and performs no side effects when constructed. Work is only submitted
//! to the GPU when the operation is *executed* inside an [`ExecutionContext`],
//! which binds it to a concrete CUDA stream.
//!
//! Operations compose through combinators (`and_then`, `zip`, `apply`) to build
//! complex dataflow graphs that remain stream-agnostic until scheduling time.
//!
//! ## Lazy scheduling
//!
//! Scheduling is the act of pairing an operation with a stream. A
//! [`SchedulingPolicy`] selects the stream (e.g., round-robin across a pool) and
//! returns a [`DeviceFuture`] that can be `.await`-ed. Until scheduled, no GPU
//! resources are consumed.
//!
//! ## DeviceFuture bridge
//!
//! [`DeviceFuture`] implements [`std::future::Future`]. On first poll it
//! executes the operation on its assigned stream, then registers a host-side
//! callback via `cuLaunchHostFunc`. The callback wakes the future through an
//! [`AtomicWaker`], yielding the result without busy-waiting.
//!
//! ## Scheduling policies
//!
//! Policies live in [`scheduling_policies`] and control stream assignment:
//!
//! | Policy                   | Behaviour                                    |
//! |--------------------------|----------------------------------------------|
//! | [`StreamPoolRoundRobin`] | Rotates across *N* streams for overlap       |
//! | [`SingleStream`]         | Serialises all work onto one stream          |
//!
//! [`DeviceOperation`]: device_operation::DeviceOperation
//! [`ExecutionContext`]: device_operation::ExecutionContext
//! [`DeviceFuture`]: device_future::DeviceFuture
//! [`SchedulingPolicy`]: scheduling_policies::SchedulingPolicy
//! [`StreamPoolRoundRobin`]: scheduling_policies::StreamPoolRoundRobin
//! [`SingleStream`]: scheduling_policies::SingleStream
//! [`AtomicWaker`]: futures::task::AtomicWaker

pub mod device_box;
pub mod device_context;
pub mod device_future;
pub mod device_operation;
pub mod error;
pub mod launch;
pub mod reclaim;
pub mod scheduling_policies;

pub use futures;
