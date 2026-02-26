-- Migration: Add trade close metadata columns to ll_predictions
-- Run this in Supabase SQL Editor if the table already exists.
-- Safe to run multiple times (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).

ALTER TABLE ll_predictions ADD COLUMN IF NOT EXISTS hold_duration_minutes DOUBLE PRECISION DEFAULT NULL;
ALTER TABLE ll_predictions ADD COLUMN IF NOT EXISTS exit_reason TEXT DEFAULT NULL;

-- Optional index for analyzing exit patterns
CREATE INDEX IF NOT EXISTS idx_ll_predictions_exit_reason ON ll_predictions(exit_reason);
