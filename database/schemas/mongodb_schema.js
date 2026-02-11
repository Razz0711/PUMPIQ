// PumpIQ MongoDB Database Schema
// Version: 1.0.0
// Database: MongoDB 6.0+

// ============================================================================
// Collection: tokens
// ============================================================================
db.createCollection("tokens", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "ticker", "contractAddress", "blockchain"],
      properties: {
        _id: {
          bsonType: "objectId"
        },
        name: {
          bsonType: "string",
          description: "Token name"
        },
        ticker: {
          bsonType: "string",
          description: "Token ticker symbol"
        },
        contractAddress: {
          bsonType: "string",
          description: "Unique contract address"
        },
        blockchain: {
          bsonType: "string",
          enum: ["solana", "ethereum", "bsc", "polygon"],
          description: "Blockchain network"
        },
        currentPrice: {
          bsonType: "double",
          description: "Current token price"
        },
        marketCap: {
          bsonType: "double",
          description: "Market capitalization"
        },
        liquidity: {
          bsonType: "double",
          description: "Total liquidity"
        },
        volume24h: {
          bsonType: "double",
          description: "24-hour trading volume"
        },
        bondingCurvePercent: {
          bsonType: "double",
          minimum: 0,
          maximum: 100,
          description: "RobinPump bonding curve percentage"
        },
        isGraduated: {
          bsonType: "bool",
          description: "Whether token has graduated from bonding curve"
        },
        graduationTimestamp: {
          bsonType: "date",
          description: "Graduation timestamp"
        },
        metadata: {
          bsonType: "object",
          properties: {
            description: { bsonType: "string" },
            website: { bsonType: "string" },
            twitter: { bsonType: "string" },
            telegram: { bsonType: "string" },
            logoUrl: { bsonType: "string" }
          }
        },
        createdAt: {
          bsonType: "date",
          description: "Record creation timestamp"
        },
        updatedAt: {
          bsonType: "date",
          description: "Last update timestamp"
        },
        lastPriceUpdate: {
          bsonType: "date",
          description: "Last price update timestamp"
        }
      }
    }
  }
});

// Indexes for tokens collection
db.tokens.createIndex({ "contractAddress": 1 }, { unique: true });
db.tokens.createIndex({ "ticker": 1 });
db.tokens.createIndex({ "blockchain": 1 });
db.tokens.createIndex({ "bondingCurvePercent": 1 });
db.tokens.createIndex({ "marketCap": -1 });
db.tokens.createIndex({ "updatedAt": -1 });

// ============================================================================
// Collection: historicalData (time-series)
// ============================================================================
db.createCollection("historicalData", {
  timeseries: {
    timeField: "timestamp",
    metaField: "tokenId",
    granularity: "minutes"
  }
});

// Sample document structure for historicalData
const historicalDataSchema = {
  _id: ObjectId(),
  tokenId: ObjectId(), // Reference to tokens collection
  timestamp: new Date(),
  
  // OHLCV data
  ohlcv: {
    open: 0.000123,
    high: 0.000156,
    low: 0.000118,
    close: 0.000145,
    volume: 125000.50
  },
  
  // On-chain metrics
  onchain: {
    holderCount: 1250,
    whaleCount: 15,
    top10HolderPercent: 45.5,
    totalTransactions: 5420,
    buyTransactions: 3200,
    sellTransactions: 2220,
    whaleActivity: {
      buys1h: 2,
      sells1h: 1,
      buyVolume1h: 15000,
      sellVolume1h: 8000
    }
  },
  
  // Sentiment scores (0-100)
  sentiment: {
    social: 72.5,
    news: 65.8
  },
  
  // Technical indicators
  technical: {
    rsi14: 58.5,
    macd: 0.000012,
    macdSignal: 0.000010,
    bollingerBands: {
      upper: 0.000165,
      middle: 0.000145,
      lower: 0.000125
    }
  }
};

// Indexes for historicalData collection
db.historicalData.createIndex({ "tokenId": 1, "timestamp": -1 });
db.historicalData.createIndex({ "timestamp": -1 });

// ============================================================================
// Collection: recommendations
// ============================================================================
db.createCollection("recommendations", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["tokenId", "recommendationType", "entryPrice", "confidenceScore", "aiReasoning"],
      properties: {
        _id: {
          bsonType: "objectId"
        },
        tokenId: {
          bsonType: "objectId",
          description: "Reference to token"
        },
        userId: {
          bsonType: "objectId",
          description: "Reference to user"
        },
        recommendationType: {
          bsonType: "string",
          enum: ["BUY", "HOLD", "SELL", "AVOID"],
          description: "Type of recommendation"
        },
        entryPrice: {
          bsonType: "double",
          description: "Recommended entry price"
        },
        currentPrice: {
          bsonType: "double",
          description: "Current price"
        },
        targets: {
          bsonType: "object",
          properties: {
            target1: { bsonType: "double" },
            target2: { bsonType: "double" },
            target3: { bsonType: "double" },
            stopLoss: { bsonType: "double" }
          }
        },
        confidenceScore: {
          bsonType: "double",
          minimum: 0,
          maximum: 1,
          description: "AI confidence score"
        },
        riskRating: {
          bsonType: "string",
          enum: ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
          description: "Risk assessment"
        },
        timeframe: {
          bsonType: "string",
          enum: ["SCALP", "DAY_TRADE", "SWING", "LONG_TERM"],
          description: "Trading timeframe"
        },
        dataContribution: {
          bsonType: "object",
          properties: {
            news: { bsonType: "bool" },
            onchain: { bsonType: "bool" },
            technical: { bsonType: "bool" },
            social: { bsonType: "bool" }
          }
        },
        aiReasoning: {
          bsonType: "string",
          description: "AI explanation for recommendation"
        },
        keyFactors: {
          bsonType: "array",
          items: {
            bsonType: "object",
            properties: {
              factor: { bsonType: "string" },
              impact: { bsonType: "string", enum: ["POSITIVE", "NEGATIVE", "NEUTRAL"] },
              weight: { bsonType: "double" }
            }
          }
        },
        status: {
          bsonType: "string",
          enum: ["ACTIVE", "COMPLETED", "STOPPED", "EXPIRED"],
          description: "Recommendation status"
        },
        createdAt: {
          bsonType: "date"
        },
        updatedAt: {
          bsonType: "date"
        },
        expiresAt: {
          bsonType: "date"
        }
      }
    }
  }
});

// Indexes for recommendations collection
db.recommendations.createIndex({ "tokenId": 1 });
db.recommendations.createIndex({ "userId": 1 });
db.recommendations.createIndex({ "status": 1 });
db.recommendations.createIndex({ "confidenceScore": -1 });
db.recommendations.createIndex({ "createdAt": -1 });
db.recommendations.createIndex({ "recommendationType": 1 });

// ============================================================================
// Collection: performanceTracking
// ============================================================================
db.createCollection("performanceTracking", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["recommendationId"],
      properties: {
        _id: {
          bsonType: "objectId"
        },
        recommendationId: {
          bsonType: "objectId",
          description: "Reference to recommendation"
        },
        outcome: {
          bsonType: "string",
          enum: ["TARGET_1_HIT", "TARGET_2_HIT", "TARGET_3_HIT", "STOP_LOSS_HIT", "STILL_OPEN", "EXPIRED"],
          description: "Final outcome"
        },
        actualExitPrice: {
          bsonType: "double",
          description: "Actual exit price"
        },
        actualReturnPercent: {
          bsonType: "double",
          description: "Actual return percentage"
        },
        timing: {
          bsonType: "object",
          properties: {
            recommendationCreatedAt: { bsonType: "date" },
            outcomeTimestamp: { bsonType: "date" },
            timeToOutcomeHours: { bsonType: "double" }
          }
        },
        success: {
          bsonType: "object",
          properties: {
            isSuccessful: { bsonType: "bool" },
            hitTarget1: { bsonType: "bool" },
            hitTarget2: { bsonType: "bool" },
            hitTarget3: { bsonType: "bool" },
            hitStopLoss: { bsonType: "bool" }
          }
        },
        priceMovement: {
          bsonType: "object",
          properties: {
            maxPriceReached: { bsonType: "double" },
            minPriceReached: { bsonType: "double" },
            maxGainPercent: { bsonType: "double" },
            maxDrawdownPercent: { bsonType: "double" }
          }
        },
        notes: {
          bsonType: "string"
        },
        createdAt: {
          bsonType: "date"
        },
        updatedAt: {
          bsonType: "date"
        }
      }
    }
  }
});

// Indexes for performanceTracking collection
db.performanceTracking.createIndex({ "recommendationId": 1 }, { unique: true });
db.performanceTracking.createIndex({ "outcome": 1 });
db.performanceTracking.createIndex({ "success.isSuccessful": 1 });
db.performanceTracking.createIndex({ "actualReturnPercent": -1 });

// ============================================================================
// Collection: userPreferences
// ============================================================================
db.createCollection("userPreferences", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["email", "config"],
      properties: {
        _id: {
          bsonType: "objectId"
        },
        username: {
          bsonType: "string"
        },
        email: {
          bsonType: "string",
          pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        },
        config: {
          bsonType: "object",
          description: "User configuration settings (matches config schema)"
        },
        watchlist: {
          bsonType: "array",
          items: {
            bsonType: "objectId"
          },
          description: "Array of token IDs"
        },
        portfolio: {
          bsonType: "array",
          items: {
            bsonType: "object",
            properties: {
              tokenId: { bsonType: "objectId" },
              quantity: { bsonType: "double" },
              avgBuyPrice: { bsonType: "double" },
              currentValue: { bsonType: "double" }
            }
          }
        },
        notificationSettings: {
          bsonType: "object"
        },
        subscription: {
          bsonType: "object",
          properties: {
            tier: {
              bsonType: "string",
              enum: ["free", "basic", "premium", "pro"]
            },
            expiresAt: { bsonType: "date" }
          }
        },
        activity: {
          bsonType: "object",
          properties: {
            lastLogin: { bsonType: "date" },
            totalRecommendationsReceived: { bsonType: "int" },
            totalTradesMade: { bsonType: "int" }
          }
        },
        createdAt: {
          bsonType: "date"
        },
        updatedAt: {
          bsonType: "date"
        }
      }
    }
  }
});

// Indexes for userPreferences collection
db.userPreferences.createIndex({ "email": 1 }, { unique: true });
db.userPreferences.createIndex({ "username": 1 }, { unique: true, sparse: true });
db.userPreferences.createIndex({ "subscription.tier": 1 });

// ============================================================================
// Collection: dataSourceCache
// ============================================================================
db.createCollection("dataSourceCache");

// Sample document structure
const cacheSchema = {
  _id: ObjectId(),
  sourceIdentifier: "news_api_crypto",
  cacheKey: "trending_tokens_2024-02-11",
  data: {
    // Arbitrary cached data
  },
  createdAt: new Date(),
  expiresAt: new Date(Date.now() + 600000), // 10 minutes
  ttlSeconds: 600,
  hitCount: 0,
  lastAccessed: new Date()
};

// Indexes for dataSourceCache collection with TTL
db.dataSourceCache.createIndex({ "sourceIdentifier": 1, "cacheKey": 1 }, { unique: true });
db.dataSourceCache.createIndex({ "expiresAt": 1 }, { expireAfterSeconds: 0 }); // TTL index
db.dataSourceCache.createIndex({ "sourceIdentifier": 1 });

// ============================================================================
// Collection: analysisLogs
// ============================================================================
db.createCollection("analysisLogs");

const analysisLogSchema = {
  _id: ObjectId(),
  tokenId: ObjectId(),
  analysisType: "AI_SYNTHESIS", // NEWS, ONCHAIN, TECHNICAL, SOCIAL, AI_SYNTHESIS
  inputData: {
    // Snapshot of input data
  },
  outputData: {
    // Analysis results
  },
  analysisResult: "Token shows strong bullish momentum...",
  performance: {
    executionTimeMs: 1250,
    tokensUsed: 1500 // For AI API calls
  },
  status: "SUCCESS", // SUCCESS, FAILED, PARTIAL
  errorMessage: null,
  timestamp: new Date()
};

// Indexes for analysisLogs collection
db.analysisLogs.createIndex({ "tokenId": 1 });
db.analysisLogs.createIndex({ "analysisType": 1 });
db.analysisLogs.createIndex({ "timestamp": -1 });
db.analysisLogs.createIndex({ "status": 1 });

// ============================================================================
// AGGREGATION PIPELINE EXAMPLES
// ============================================================================

// Example 1: Get top performing recommendations from last 7 days
const topPerformers = db.performanceTracking.aggregate([
  {
    $match: {
      "success.isSuccessful": true,
      "timing.outcomeTimestamp": {
        $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
      }
    }
  },
  {
    $lookup: {
      from: "recommendations",
      localField: "recommendationId",
      foreignField: "_id",
      as: "recommendation"
    }
  },
  {
    $unwind: "$recommendation"
  },
  {
    $lookup: {
      from: "tokens",
      localField: "recommendation.tokenId",
      foreignField: "_id",
      as: "token"
    }
  },
  {
    $unwind: "$token"
  },
  {
    $sort: { actualReturnPercent: -1 }
  },
  {
    $limit: 10
  },
  {
    $project: {
      tokenName: "$token.name",
      ticker: "$token.ticker",
      returnPercent: "$actualReturnPercent",
      outcome: "$outcome",
      timeToOutcome: "$timing.timeToOutcomeHours"
    }
  }
]);

// Example 2: Get user's watchlist with current prices
const watchlistWithPrices = db.userPreferences.aggregate([
  {
    $match: { email: "user@example.com" }
  },
  {
    $lookup: {
      from: "tokens",
      localField: "watchlist",
      foreignField: "_id",
      as: "watchlistTokens"
    }
  },
  {
    $unwind: "$watchlistTokens"
  },
  {
    $project: {
      tokenName: "$watchlistTokens.name",
      ticker: "$watchlistTokens.ticker",
      currentPrice: "$watchlistTokens.currentPrice",
      marketCap: "$watchlistTokens.marketCap",
      bondingCurve: "$watchlistTokens.bondingCurvePercent"
    }
  }
]);

// Example 3: Calculate AI model accuracy over time
const modelAccuracy = db.performanceTracking.aggregate([
  {
    $match: {
      "timing.outcomeTimestamp": { $ne: null }
    }
  },
  {
    $group: {
      _id: {
        $dateToString: {
          format: "%Y-%m-%d",
          date: "$timing.outcomeTimestamp"
        }
      },
      totalRecommendations: { $sum: 1 },
      successful: {
        $sum: {
          $cond: ["$success.isSuccessful", 1, 0]
        }
      },
      avgReturn: { $avg: "$actualReturnPercent" }
    }
  },
  {
    $project: {
      date: "$_id",
      totalRecommendations: 1,
      successful: 1,
      winRate: {
        $multiply: [
          { $divide: ["$successful", "$totalRecommendations"] },
          100
        ]
      },
      avgReturn: { $round: ["$avgReturn", 2] }
    }
  },
  {
    $sort: { date: -1 }
  }
]);

// Example 4: Get tokens in preferred bonding curve range
const preferredTokens = db.tokens.find({
  bondingCurvePercent: { $gte: 60, $lte: 85 },
  liquidity: { $gt: 50000 },
  isGraduated: false
}).sort({ bondingCurvePercent: -1 });

// Example 5: Get recent active recommendations with performance
const activeRecommendationsWithPerformance = db.recommendations.aggregate([
  {
    $match: { status: "ACTIVE" }
  },
  {
    $lookup: {
      from: "tokens",
      localField: "tokenId",
      foreignField: "_id",
      as: "token"
    }
  },
  {
    $unwind: "$token"
  },
  {
    $addFields: {
      currentReturnPercent: {
        $cond: {
          if: { $ne: ["$currentPrice", null] },
          then: {
            $multiply: [
              {
                $divide: [
                  { $subtract: ["$currentPrice", "$entryPrice"] },
                  "$entryPrice"
                ]
              },
              100
            ]
          },
          else: null
        }
      }
    }
  },
  {
    $project: {
      tokenName: "$token.name",
      ticker: "$token.ticker",
      recommendationType: 1,
      entryPrice: 1,
      currentPrice: 1,
      currentReturnPercent: { $round: ["$currentReturnPercent", 2] },
      targets: 1,
      confidenceScore: 1,
      riskRating: 1,
      aiReasoning: 1,
      createdAt: 1
    }
  },
  {
    $sort: { createdAt: -1 }
  }
]);
