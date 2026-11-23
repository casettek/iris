//! The commands of the `iris` binary: for each group, its arguments and
//! what it does with them. `crate::execute` dispatches here.

pub(crate) mod challenge;
pub(crate) mod claim;
pub(crate) mod da;
pub(crate) mod devstack;
pub(crate) mod evm;
pub(crate) mod fault;
pub(crate) mod infer;
pub(crate) mod model;
pub(crate) mod registry;
pub(crate) mod stage;
