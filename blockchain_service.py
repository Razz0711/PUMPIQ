"""
PumpIQ Blockchain Service
============================================================
Handles on-chain recording and verification of transactions
on Base (L2) and Ethereum via the TransactionRegistry smart contract.

Architecture:
  PumpIQ Backend ──→ blockchain_service.py ──→ Base/Ethereum RPC
                                                    │
                                              TransactionRegistry.sol
                                              (on-chain tx records)

Every trade (buy/sell/deposit/withdraw) gets:
  1. A local SHA-256 hash (instant, stored in SQLite)
  2. An on-chain record via the smart contract (async, stored on Base/Ethereum)
  3. A real blockchain transaction hash (the on-chain tx receipt)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Contract ABI (only the functions we call) ──────────────────
CONTRACT_ABI = [
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
        "inputs": [],
        "name": "getTransactionCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
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
        "inputs": [],
        "name": "owner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
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
]

# ── Transaction Type Mapping ───────────────────────────────────
TX_TYPES = {
    "BUY": 0,
    "SELL": 1,
    "deposit": 2,
    "withdraw": 3,
    "signup_bonus": 4,
}

# ── Network Configuration ─────────────────────────────────────
NETWORKS = {
    "base_sepolia": {
        "rpc": "https://sepolia.base.org",
        "chain_id": 84532,
        "explorer": "https://sepolia.basescan.org",
        "name": "Base Sepolia Testnet",
    },
    "base_mainnet": {
        "rpc": "https://mainnet.base.org",
        "chain_id": 8453,
        "explorer": "https://basescan.org",
        "name": "Base Mainnet",
    },
    "ethereum": {
        "rpc": "https://eth.llamarpc.com",
        "chain_id": 1,
        "explorer": "https://etherscan.io",
        "name": "Ethereum Mainnet",
    },
}


class BlockchainService:
    """
    Service for recording PumpIQ transactions on Base/Ethereum blockchain.
    Falls back gracefully when blockchain is not configured (local-only hashes still work).
    """

    def __init__(self):
        self.enabled = False
        self.w3 = None
        self.contract = None
        self.account = None
        self.network_name = ""
        self.explorer_url = ""
        self.chain_id = 0
        self._nonce_lock = threading.Lock()

        self._initialize()

    def _initialize(self):
        """Try to connect to blockchain. If not configured, fall back silently."""
        private_key = os.getenv("DEPLOYER_PRIVATE_KEY", "").strip()
        contract_address = os.getenv("CONTRACT_ADDRESS", "").strip()
        network = os.getenv("BLOCKCHAIN_NETWORK", "base_sepolia").strip()

        if not private_key or not contract_address:
            logger.info("Blockchain not configured — running in local-hash-only mode. "
                        "Set DEPLOYER_PRIVATE_KEY and CONTRACT_ADDRESS in .env to enable on-chain recording.")
            return

        net_config = NETWORKS.get(network)
        if not net_config:
            logger.warning(f"Unknown blockchain network: {network}")
            return

        rpc_url = os.getenv(f"{network.upper()}_RPC", net_config["rpc"])

        try:
            from web3 import Web3
            self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 15}))

            if not self.w3.is_connected():
                logger.warning(f"Cannot connect to {network} at {rpc_url}")
                return

            self.account = self.w3.eth.account.from_key(private_key)
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=CONTRACT_ABI,
            )
            self.chain_id = net_config["chain_id"]
            self.network_name = net_config["name"]
            self.explorer_url = net_config["explorer"]
            self.enabled = True

            balance = self.w3.eth.get_balance(self.account.address)
            balance_eth = self.w3.from_wei(balance, "ether")
            logger.info(f"⛓ Blockchain connected: {self.network_name}")
            logger.info(f"  Contract: {contract_address}")
            logger.info(f"  Wallet: {self.account.address} ({balance_eth:.6f} ETH)")

        except ImportError:
            logger.warning("web3 package not installed. Run: pip install web3")
        except Exception as e:
            logger.warning(f"Blockchain init failed: {e}")

    def is_configured(self) -> bool:
        return self.enabled

    def get_status(self) -> Dict[str, Any]:
        """Return blockchain connection status for health check."""
        if not self.enabled:
            return {
                "enabled": False,
                "network": None,
                "message": "Not configured — set DEPLOYER_PRIVATE_KEY and CONTRACT_ADDRESS in .env",
            }

        try:
            balance = self.w3.eth.get_balance(self.account.address)
            balance_eth = float(self.w3.from_wei(balance, "ether"))
            tx_count = self.contract.functions.totalTransactions().call()
            return {
                "enabled": True,
                "network": self.network_name,
                "chain_id": self.chain_id,
                "contract_address": self.contract.address,
                "wallet_address": self.account.address,
                "wallet_balance_eth": round(balance_eth, 6),
                "on_chain_tx_count": tx_count,
                "explorer": self.explorer_url,
            }
        except Exception as e:
            return {"enabled": True, "network": self.network_name, "error": str(e)}

    def record_transaction(
        self,
        tx_hash_hex: str,
        tx_type: str,
        symbol: str,
        amount_usd: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Record a transaction hash on the blockchain.

        Args:
            tx_hash_hex: The SHA-256 hex digest (64 chars) computed locally
            tx_type: "BUY", "SELL", "deposit", "withdraw", "signup_bonus"
            symbol: Token symbol (e.g. "BTC", "ETH") or "USD" for wallet txns
            amount_usd: Dollar amount of the transaction

        Returns:
            Dict with on-chain tx hash and explorer link, or None if failed
        """
        if not self.enabled:
            return None

        try:
            # Convert hex string to bytes32
            tx_hash_bytes = bytes.fromhex(tx_hash_hex)

            # Map type string to uint8
            type_id = TX_TYPES.get(tx_type, 0)

            # Convert USD to cents (uint256)
            amount_cents = int(amount_usd * 100)

            # Build transaction
            with self._nonce_lock:
                nonce = self.w3.eth.get_transaction_count(self.account.address)

                tx = self.contract.functions.recordTransaction(
                    tx_hash_bytes,
                    type_id,
                    symbol.upper(),
                    amount_cents,
                ).build_transaction({
                    "chainId": self.chain_id,
                    "from": self.account.address,
                    "nonce": nonce,
                    "gasPrice": self.w3.eth.gas_price,
                })

                gas_estimate = self.w3.eth.estimate_gas(tx)
                tx["gas"] = int(gas_estimate * 1.2)

                signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
                on_chain_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

            # Wait for receipt (with timeout)
            receipt = self.w3.eth.wait_for_transaction_receipt(on_chain_hash, timeout=30)

            result = {
                "on_chain_tx_hash": on_chain_hash.hex(),
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "status": "confirmed" if receipt.status == 1 else "failed",
                "explorer_url": f"{self.explorer_url}/tx/0x{on_chain_hash.hex()}",
                "network": self.network_name,
            }

            logger.info(f"⛓ On-chain: {tx_type} {symbol} ${amount_usd:.2f} → {result['explorer_url']}")
            return result

        except Exception as e:
            logger.warning(f"On-chain recording failed (local hash still valid): {e}")
            return None

    def record_transaction_async(
        self,
        tx_hash_hex: str,
        tx_type: str,
        symbol: str,
        amount_usd: float,
        callback=None,
    ):
        """
        Record transaction on-chain in a background thread (non-blocking).
        The local SHA-256 hash is already saved to SQLite instantly.
        The on-chain recording happens asynchronously.
        """
        if not self.enabled:
            return

        def _background():
            result = self.record_transaction(tx_hash_hex, tx_type, symbol, amount_usd)
            if callback and result:
                callback(tx_hash_hex, result)

        thread = threading.Thread(target=_background, daemon=True)
        thread.start()

    def verify_on_chain(self, tx_hash_hex: str) -> Optional[Dict[str, Any]]:
        """
        Verify a transaction exists on the blockchain.

        Returns:
            Dict with on-chain record data, or None if not found/not configured
        """
        if not self.enabled:
            return None

        try:
            tx_hash_bytes = bytes.fromhex(tx_hash_hex)
            result = self.contract.functions.verifyTransaction(tx_hash_bytes).call()
            exists, recorder, tx_type, symbol, amount_cents, timestamp = result

            if not exists:
                return {"exists": False, "network": self.network_name}

            type_names = {0: "BUY", 1: "SELL", 2: "DEPOSIT", 3: "WITHDRAW", 4: "BONUS"}
            return {
                "exists": True,
                "network": self.network_name,
                "recorder": recorder,
                "tx_type": type_names.get(tx_type, f"UNKNOWN({tx_type})"),
                "symbol": symbol,
                "amount_usd": amount_cents / 100,
                "timestamp": timestamp,
                "explorer_url": f"{self.explorer_url}/address/{self.contract.address}",
            }

        except Exception as e:
            logger.warning(f"On-chain verify failed: {e}")
            return None

    def get_on_chain_count(self) -> int:
        """Get total number of transactions recorded on-chain."""
        if not self.enabled:
            return 0
        try:
            return self.contract.functions.totalTransactions().call()
        except Exception:
            return 0


# ── Singleton instance ─────────────────────────────────────────
blockchain = BlockchainService()
