"""
PumpIQ — Deploy TransactionRegistry to Base Sepolia (or Base/Ethereum mainnet).

Prerequisites:
  1. pip install web3 py-solc-x
  2. Get free testnet ETH: https://www.coinbase.com/faucets/base-ethereum-goerli-faucet
  3. Set in .env:
       DEPLOYER_PRIVATE_KEY=0x...   (your MetaMask private key)
       BASE_SEPOLIA_RPC=https://sepolia.base.org

Usage:
  python contracts/deploy.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# ── Configuration ──────────────────────────────────────────────
NETWORK = os.getenv("BLOCKCHAIN_NETWORK", "base_sepolia")

RPC_URLS = {
    "base_sepolia": os.getenv("BASE_SEPOLIA_RPC", "https://sepolia.base.org"),
    "base_mainnet": os.getenv("BASE_MAINNET_RPC", "https://mainnet.base.org"),
    "ethereum":     os.getenv("ETHEREUM_RPC", "https://eth.llamarpc.com"),
}

CHAIN_IDS = {
    "base_sepolia": 84532,
    "base_mainnet": 8453,
    "ethereum":     1,
}

EXPLORER_URLS = {
    "base_sepolia": "https://sepolia.basescan.org",
    "base_mainnet": "https://basescan.org",
    "ethereum":     "https://etherscan.io",
}

PRIVATE_KEY = os.getenv("DEPLOYER_PRIVATE_KEY", "")

# ── Pre-compiled contract ABI & Bytecode ───────────────────────
# This is the compiled output of TransactionRegistry.sol
# Compiled with solc 0.8.20, optimizer enabled (200 runs)
# You can recompile with: solcx.compile_source(source, ...)

# ABI (Application Binary Interface)
CONTRACT_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "txHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "recorder", "type": "address"},
            {"indexed": False, "internalType": "uint8", "name": "txType", "type": "uint8"},
            {"indexed": False, "internalType": "string", "name": "symbol", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "TransactionRecorded",
        "type": "event"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "_txHash", "type": "bytes32"},
            {"internalType": "uint8", "name": "_txType", "type": "uint8"},
            {"internalType": "string", "name": "_symbol", "type": "string"},
            {"internalType": "uint256", "name": "_amount", "type": "uint256"}
        ],
        "name": "recordTransaction",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32[]", "name": "_txHashes", "type": "bytes32[]"},
            {"internalType": "uint8[]", "name": "_txTypes", "type": "uint8[]"},
            {"internalType": "string[]", "name": "_symbols", "type": "string[]"},
            {"internalType": "uint256[]", "name": "_amounts", "type": "uint256[]"}
        ],
        "name": "batchRecordTransactions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "_txHash", "type": "bytes32"}],
        "name": "verifyTransaction",
        "outputs": [
            {"internalType": "bool", "name": "exists_", "type": "bool"},
            {"internalType": "address", "name": "recorder_", "type": "address"},
            {"internalType": "uint8", "name": "txType_", "type": "uint8"},
            {"internalType": "string", "name": "symbol_", "type": "string"},
            {"internalType": "uint256", "name": "amount_", "type": "uint256"},
            {"internalType": "uint256", "name": "timestamp_", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "_user", "type": "address"}],
        "name": "getUserTransactions",
        "outputs": [{"internalType": "bytes32[]", "name": "", "type": "bytes32[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getTransactionCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalTransactions",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "transactionHashes",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "name": "transactions",
        "outputs": [
            {"internalType": "bytes32", "name": "txHash", "type": "bytes32"},
            {"internalType": "address", "name": "recorder", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "uint8", "name": "txType", "type": "uint8"},
            {"internalType": "string", "name": "symbol", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "bool", "name": "exists", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "_newOwner", "type": "address"}],
        "name": "transferOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "name": "userTransactions",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    }
]


def deploy():
    if not PRIVATE_KEY:
        print("❌ Set DEPLOYER_PRIVATE_KEY in .env (your MetaMask wallet private key)")
        print("   How to get it:")
        print("   1. Open MetaMask → click ⋮ → Account Details → Show Private Key")
        print("   2. Copy it and add to .env: DEPLOYER_PRIVATE_KEY=0x...")
        print("\n   Also get free Base Sepolia ETH from:")
        print("   https://www.coinbase.com/faucets/base-ethereum-goerli-faucet")
        sys.exit(1)

    rpc_url = RPC_URLS.get(NETWORK)
    chain_id = CHAIN_IDS.get(NETWORK)
    explorer = EXPLORER_URLS.get(NETWORK)

    print(f"🔗 Network: {NETWORK}")
    print(f"🌐 RPC: {rpc_url}")
    print(f"🔢 Chain ID: {chain_id}")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"❌ Cannot connect to {rpc_url}")
        sys.exit(1)
    print(f"✅ Connected to {NETWORK}")

    account = w3.eth.account.from_key(PRIVATE_KEY)
    deployer = account.address
    balance = w3.eth.get_balance(deployer)
    balance_eth = w3.from_wei(balance, "ether")
    print(f"👤 Deployer: {deployer}")
    print(f"💰 Balance: {balance_eth} ETH")

    if balance == 0:
        print("❌ No ETH! Get free testnet ETH from the faucet first.")
        sys.exit(1)

    # Compile the contract using py-solc-x
    try:
        import solcx
        solcx.install_solc("0.8.20")
        sol_path = os.path.join(os.path.dirname(__file__), "TransactionRegistry.sol")
        with open(sol_path, "r") as f:
            source = f.read()
        compiled = solcx.compile_source(
            source,
            output_values=["abi", "bin"],
            solc_version="0.8.20",
        )
        contract_id, contract_interface = compiled.popitem()
        abi = contract_interface["abi"]
        bytecode = contract_interface["bin"]
        print("✅ Contract compiled successfully")
    except ImportError:
        print("⚠️  py-solc-x not installed. Install with: pip install py-solc-x")
        print("   Falling back to pre-compiled bytecode...")
        # If py-solc-x is not available, you need to paste the compiled bytecode here
        # You can compile at https://remix.ethereum.org and paste the bytecode
        print("❌ Please compile the contract at https://remix.ethereum.org")
        print("   1. Paste TransactionRegistry.sol in Remix")
        print("   2. Compile with Solidity 0.8.20")
        print("   3. Copy the bytecode and ABI")
        print("   4. Or install py-solc-x: pip install py-solc-x")
        sys.exit(1)

    # Deploy
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(deployer)

    tx = Contract.constructor().build_transaction({
        "chainId": chain_id,
        "from": deployer,
        "nonce": nonce,
        "gasPrice": w3.eth.gas_price,
    })

    # Estimate gas
    gas_estimate = w3.eth.estimate_gas(tx)
    tx["gas"] = int(gas_estimate * 1.2)  # 20% buffer

    print(f"\n🚀 Deploying TransactionRegistry...")
    print(f"   Estimated gas: {gas_estimate:,}")

    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"   TX Hash: {tx_hash.hex()}")
    print(f"   Waiting for confirmation...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    contract_address = receipt.contractAddress

    print(f"\n{'='*60}")
    print(f"✅ CONTRACT DEPLOYED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"   Contract Address: {contract_address}")
    print(f"   TX Hash:          {tx_hash.hex()}")
    print(f"   Block:            {receipt.blockNumber}")
    print(f"   Gas Used:         {receipt.gasUsed:,}")
    print(f"   Explorer:         {explorer}/address/{contract_address}")
    print(f"\n📝 Add to your .env file:")
    print(f"   CONTRACT_ADDRESS={contract_address}")

    # Save deployment info
    deployment_info = {
        "network": NETWORK,
        "contract_address": contract_address,
        "deployer": deployer,
        "tx_hash": tx_hash.hex(),
        "block_number": receipt.blockNumber,
        "gas_used": receipt.gasUsed,
        "chain_id": chain_id,
        "explorer_url": f"{explorer}/address/{contract_address}",
        "abi": abi,
    }
    info_path = os.path.join(os.path.dirname(__file__), "deployment.json")
    with open(info_path, "w") as f:
        json.dump(deployment_info, f, indent=2)
    print(f"\n💾 Deployment info saved to contracts/deployment.json")


if __name__ == "__main__":
    deploy()
