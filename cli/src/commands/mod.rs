//! The commands of the `iwebs` binary: for each group, its arguments and
//! what it does with them. `crate::execute` dispatches here.

pub(crate) mod challenge;
pub(crate) mod claim;
pub(crate) mod da;
pub(crate) mod devstack;
pub(crate) mod evm;
pub(crate) mod fault;
pub(crate) mod infer;
pub(crate) mod model;
pub(crate) mod procedure;
pub(crate) mod record;
pub(crate) mod registry;
pub(crate) mod run;
pub(crate) mod stage;
pub(crate) mod testing;
