// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title PumpIQ Transaction Registry
 * @notice Records transaction hashes on-chain for verifiable trading history.
 *         Deployed first on Base Sepolia (testnet), then Base mainnet, then Ethereum.
 * @dev    Each transaction stores a SHA-256 digest computed off-chain plus metadata.
 *         Anyone can verify a transaction's existence and integrity on-chain.
 */
contract TransactionRegistry {

    // ── Types ─────────────────────────────────────────────────
    struct TxRecord {
        bytes32 txHash;        // SHA-256 digest of trade data
        address recorder;      // wallet that submitted the record
        uint256 amount;        // trade amount in cents (USD × 100)
        uint8   txType;        // 0 = BUY, 1 = SELL, 2 = DEPOSIT, 3 = WITHDRAW, 4 = BONUS
        string  symbol;        // e.g. "BTC", "ETH", "SOL"
        uint256 timestamp;     // block timestamp when recorded
        bool    exists;        // flag for existence check
    }

    // ── State ─────────────────────────────────────────────────
    address public owner;
    uint256 public totalTransactions;

    mapping(bytes32 => TxRecord) public transactions;   // txHash → record
    bytes32[] public transactionHashes;                  // ordered list

    // Per-user transaction lists
    mapping(address => bytes32[]) public userTransactions;

    // ── Events ────────────────────────────────────────────────
    event TransactionRecorded(
        bytes32 indexed txHash,
        address indexed recorder,
        uint8   txType,
        string  symbol,
        uint256 amount,
        uint256 timestamp
    );

    // ── Modifiers ─────────────────────────────────────────────
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    // ── Constructor ───────────────────────────────────────────
    constructor() {
        owner = msg.sender;
    }

    // ── Core Functions ────────────────────────────────────────

    /**
     * @notice Record a transaction hash on-chain.
     * @param _txHash   SHA-256 hash of the transaction data (computed off-chain)
     * @param _txType   0=BUY, 1=SELL, 2=DEPOSIT, 3=WITHDRAW, 4=BONUS
     * @param _symbol   Token symbol (e.g. "BTC")
     * @param _amount   Amount in USD cents (e.g. $150.50 = 15050)
     */
    function recordTransaction(
        bytes32 _txHash,
        uint8   _txType,
        string calldata _symbol,
        uint256 _amount
    ) external {
        require(!transactions[_txHash].exists, "Transaction already recorded");
        require(_txType <= 4, "Invalid transaction type");

        TxRecord memory record = TxRecord({
            txHash:    _txHash,
            recorder:  msg.sender,
            amount:    _amount,
            txType:    _txType,
            symbol:    _symbol,
            timestamp: block.timestamp,
            exists:    true
        });

        transactions[_txHash] = record;
        transactionHashes.push(_txHash);
        userTransactions[msg.sender].push(_txHash);
        totalTransactions++;

        emit TransactionRecorded(_txHash, msg.sender, _txType, _symbol, _amount, block.timestamp);
    }

    /**
     * @notice Verify a transaction exists on-chain and return its data.
     */
    function verifyTransaction(bytes32 _txHash) external view returns (
        bool    exists_,
        address recorder_,
        uint8   txType_,
        string memory symbol_,
        uint256 amount_,
        uint256 timestamp_
    ) {
        TxRecord memory r = transactions[_txHash];
        return (r.exists, r.recorder, r.txType, r.symbol, r.amount, r.timestamp);
    }

    /**
     * @notice Get all transaction hashes for a specific user/wallet.
     */
    function getUserTransactions(address _user) external view returns (bytes32[] memory) {
        return userTransactions[_user];
    }

    /**
     * @notice Get the total count of recorded transactions.
     */
    function getTransactionCount() external view returns (uint256) {
        return totalTransactions;
    }

    /**
     * @notice Batch-record multiple transactions in one call (saves gas).
     */
    function batchRecordTransactions(
        bytes32[] calldata _txHashes,
        uint8[]   calldata _txTypes,
        string[]  calldata _symbols,
        uint256[] calldata _amounts
    ) external {
        require(
            _txHashes.length == _txTypes.length &&
            _txHashes.length == _symbols.length &&
            _txHashes.length == _amounts.length,
            "Array length mismatch"
        );

        for (uint256 i = 0; i < _txHashes.length; i++) {
            if (!transactions[_txHashes[i]].exists && _txTypes[i] <= 4) {
                TxRecord memory record = TxRecord({
                    txHash:    _txHashes[i],
                    recorder:  msg.sender,
                    amount:    _amounts[i],
                    txType:    _txTypes[i],
                    symbol:    _symbols[i],
                    timestamp: block.timestamp,
                    exists:    true
                });

                transactions[_txHashes[i]] = record;
                transactionHashes.push(_txHashes[i]);
                userTransactions[msg.sender].push(_txHashes[i]);
                totalTransactions++;

                emit TransactionRecorded(
                    _txHashes[i], msg.sender, _txTypes[i],
                    _symbols[i], _amounts[i], block.timestamp
                );
            }
        }
    }

    /**
     * @notice Transfer ownership (for multi-sig or DAO governance later).
     */
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "Invalid address");
        owner = _newOwner;
    }
}
