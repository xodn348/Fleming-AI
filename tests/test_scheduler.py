"""
Tests for Fleming-AI scheduler runner
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.scheduler.runner import FlemingRunner


class TestFlemingRunner:
    """Test suite for FlemingRunner"""

    @pytest.fixture
    def runner(self):
        """Create a runner instance with short delays for testing"""
        return FlemingRunner(cycle_delay=1, max_retries=2, retry_delay=0.1)

    @pytest.mark.asyncio
    async def test_runner_initialization(self, runner):
        """Test runner initializes with correct parameters"""
        assert runner.cycle_delay == 1
        assert runner.max_retries == 2
        assert runner.retry_delay == 0.1
        assert runner._running is False

    @pytest.mark.asyncio
    async def test_run_once_success(self, runner):
        """Test successful single cycle execution"""
        # Mock all pipeline steps
        runner._collect_papers = AsyncMock()
        runner._generate_hypotheses = AsyncMock()
        runner._validate_hypotheses = AsyncMock()
        runner._store_results = AsyncMock()
        runner._sync_data = AsyncMock()

        result = await runner.run_once()

        assert result is True
        runner._collect_papers.assert_called_once()
        runner._generate_hypotheses.assert_called_once()
        runner._validate_hypotheses.assert_called_once()
        runner._store_results.assert_called_once()
        runner._sync_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_once_failure(self, runner):
        """Test cycle failure handling"""
        # Mock failure in collect step
        runner._collect_papers = AsyncMock(side_effect=Exception("Collection failed"))

        result = await runner.run_once()

        assert result is False

    @pytest.mark.asyncio
    async def test_run_once_partial_failure(self, runner):
        """Test failure in middle of cycle"""
        runner._collect_papers = AsyncMock()
        runner._generate_hypotheses = AsyncMock()
        runner._validate_hypotheses = AsyncMock(side_effect=Exception("Validation failed"))

        result = await runner.run_once()

        assert result is False
        # First two steps should have been called
        runner._collect_papers.assert_called_once()
        runner._generate_hypotheses.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_signal(self, runner):
        """Test graceful shutdown signal"""
        assert runner._running is False
        runner._running = True

        runner.stop()

        assert runner._running is False
        assert runner._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup(self, runner):
        """Test cleanup completes without error"""
        await runner.cleanup()
        # Should complete without raising

    @pytest.mark.asyncio
    async def test_run_forever_with_immediate_stop(self, runner):
        """Test run_forever stops when signaled"""
        # Mock successful cycle
        runner._collect_papers = AsyncMock()
        runner._generate_hypotheses = AsyncMock()
        runner._validate_hypotheses = AsyncMock()
        runner._store_results = AsyncMock()
        runner._sync_data = AsyncMock()

        # Stop after first cycle
        async def stop_after_delay():
            await asyncio.sleep(0.5)
            runner.stop()

        # Run both tasks
        await asyncio.gather(runner.run_forever(), stop_after_delay())

        assert runner._running is False

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, runner):
        """Test retry mechanism on cycle failure"""
        # Fail first time, succeed second time
        call_count = 0

        async def mock_collect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt failed")

        runner._collect_papers = mock_collect
        runner._generate_hypotheses = AsyncMock()
        runner._validate_hypotheses = AsyncMock()
        runner._store_results = AsyncMock()
        runner._sync_data = AsyncMock()

        # Stop after retries
        async def stop_after_delay():
            await asyncio.sleep(1)
            runner.stop()

        await asyncio.gather(runner.run_forever(), stop_after_delay())

        # Should have retried
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, runner):
        """Test behavior when max retries exceeded"""
        # Always fail
        runner._collect_papers = AsyncMock(side_effect=Exception("Always fails"))

        # Stop after a short time
        async def stop_after_delay():
            await asyncio.sleep(0.5)
            runner.stop()

        await asyncio.gather(runner.run_forever(), stop_after_delay())

        # Should have attempted max_retries times
        assert runner._collect_papers.call_count >= runner.max_retries


class TestFlemingRunnerIntegration:
    """Integration tests for runner with real async operations"""

    @pytest.mark.asyncio
    async def test_full_cycle_with_delays(self):
        """Test full cycle with realistic async delays"""
        runner = FlemingRunner(cycle_delay=0.1, max_retries=1, retry_delay=0.05)

        # Mock with small delays to simulate real work
        async def mock_step():
            await asyncio.sleep(0.01)

        runner._collect_papers = mock_step
        runner._generate_hypotheses = mock_step
        runner._validate_hypotheses = mock_step
        runner._store_results = mock_step
        runner._sync_data = mock_step

        result = await runner.run_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_concurrent_shutdown(self):
        """Test shutdown during cycle execution"""
        runner = FlemingRunner(cycle_delay=0.1)

        # Mock with longer delays
        async def slow_step():
            await asyncio.sleep(0.5)

        runner._collect_papers = slow_step
        runner._generate_hypotheses = AsyncMock()
        runner._validate_hypotheses = AsyncMock()
        runner._store_results = AsyncMock()
        runner._sync_data = AsyncMock()

        # Stop during execution
        async def stop_immediately():
            await asyncio.sleep(0.05)
            runner.stop()

        await asyncio.gather(runner.run_forever(), stop_immediately())

        assert runner._running is False
