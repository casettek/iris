//! The code that needs a blockchain: Celestia as the layer a claim's data is
//! posted to, the calls IWEBS makes to its EVM contracts, and the local
//! blockchains as a test drives them.

pub mod celestia;
pub mod devstack;
pub mod evm;
