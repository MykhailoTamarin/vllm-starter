import asyncio
import subprocess
import time
import sys
from datetime import datetime, timezone
from typing import List
import aiohttp

from ._version import __version__
from .config import BenchmarkConfig
from .client import LLMClient
from .prompts import PromptGenerator
from .results import BenchmarkResults, BenchmarkMetadata

class BenchmarkFailure(Exception):
    pass

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, client: LLMClient, prompt_generator: PromptGenerator):
        self.config = config
        self.client = client
        self.prompt_gen = prompt_generator
        self.results = BenchmarkResults()

        # We need to track deltas from warmup to adapt prompts
        self.delta_user = 0
        self.delta_context = 0

    async def wait_for_idle(self, session: aiohttp.ClientSession):
        """Poll /metrics until num_requests_running=0 and num_requests_waiting=0."""
        self._idle_retries = 0
        idle_interval = self.config.idle_interval
        max_retries = self.config.idle_max_retries
        start_time = time.monotonic()
        # metrics is at root path, /v1 is OpenAI API prefix only
        from urllib.parse import urlparse
        parsed = urlparse(self.client.base_url)
        host_and_port = f"{parsed.scheme}://{parsed.netloc}"
        metrics_url = host_and_port + '/metrics'
        print("\n  ⏳ Waiting for idle (grace period 5s)...")
        sys.stderr.flush()

        # Grace period: let spec-decode verification and server-side cleanup complete
        await asyncio.sleep(5)

        timeout_s = idle_interval * max_retries
        while time.monotonic() - start_time < timeout_s:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get(metrics_url, timeout=timeout) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        running = 0
                        waiting = 0
                        for line in text.splitlines():
                            stripped = line.strip()
                            if stripped.startswith("#"):
                                continue
                            if stripped.startswith("vllm:num_requests_running"):
                                parts = stripped.split()
                                if len(parts) >= 2:
                                    try:
                                        running = int(float(parts[-1]))
                                    except Exception:
                                        pass
                            elif stripped.startswith("vllm:num_requests_waiting"):
                                parts = stripped.split()
                                if len(parts) >= 2:
                                    try:
                                        waiting = int(float(parts[-1]))
                                    except Exception:
                                        pass

                        if running == 0 and waiting == 0:
                            print(f"  ✅ Idle: Running: {running}, Waiting: {waiting}")
                            return
                        
                        print(f"  ⏳ Running: {running}, Waiting: {waiting}...")
                    else:
                        print(f"\n  ⚠️ Metrics returned {resp.status}, retrying...")
            except Exception as e:
                if self._idle_retries == 0:
                    print(f"\n  ⚠️ Metrics error: {e}, retrying...")
                pass
            
            self._idle_retries += 1
            await asyncio.sleep(idle_interval)

        print(f"\n  ⚠️ Idle wait timeout after {timeout_s}s — continuing anyway")

    async def run_suite(self):
        # Initialize session
        timeout = aiohttp.ClientTimeout(total=3600)
        max_concurrency = max(self.config.concurrency_levels)
        connector = aiohttp.TCPConnector(limit=max_concurrency + 5, force_close=False, keepalive_timeout=600)
        latency = 0.0  # default in case of early interrupt

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
                # Warmup
                should_warmup = not self.config.no_warmup
                if self.config.adapt_prompt:
                    should_warmup = True

                tokenizer = self.prompt_gen.corpus.get_tokenizer()

                if should_warmup:
                    self.delta_user, self.delta_context = await self.client.warmup(session, tokenizer)

                # Coherence test after warmup (by default, unless skipped)
                if not self.config.skip_coherence:
                    if not await self.client.run_coherence_test(session):
                        print("\nBenchmark failed due to coherence test failure.")
                        raise SystemExit(1)
                else:
                    print("\nSkipping coherence test (--skip-coherence specified)")

                # Measure latency
                latency = await self.client.measure_latency(session, self.config.latency_mode)

                # Wait for idle after warmup + coherence test
                if self.config.idle_wait:
                    await self.wait_for_idle(session)

                # Main Loop
                for depth in self.config.depths:
                    for pp in self.config.pp_counts:
                        for tg in self.config.tg_counts:
                            for concurrency in self.config.concurrency_levels:
                                print(f"Running test: pp={pp}, tg={tg}, depth={depth}, concurrency={concurrency}")

                                run_std_results = []
                                run_ctx_results = []
                                expected_pp = pp
                                expected_ctx = depth

                                total_runs = self.config.num_runs if self.config.no_warmup else self.config.num_runs + 1
                                for run in range(total_runs):
                                    is_warmup = not self.config.no_warmup and run == 0
                                    run_label = "Warmup" if is_warmup else f"Run {run if not self.config.no_warmup else run + 1}/{self.config.num_runs}"

                                    # Adapt prompt tokens
                                    current_pp = pp
                                    current_depth = depth
                                    if self.config.adapt_prompt:
                                        if depth == 0:
                                            current_pp = max(1, pp - self.delta_user)
                                        else:
                                            current_depth = max(1, depth - self.delta_context)

                                    expected_pp = current_pp
                                    expected_ctx = current_depth

                                    prompt_batch = self.prompt_gen.generate_batch(
                                        concurrency,
                                        current_pp,
                                        current_depth,
                                        self.config.no_cache
                                    )

                                    if self.config.enable_prefix_caching and depth > 0:
                                        # Phase 1: Context Load
                                        print(f"  {run_label} (Context Load, batch size {concurrency})...")
                                        load_tasks = []
                                        for i in range(concurrency):
                                            context, _ = prompt_batch[i]
                                            load_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text="",
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer
                                            ))

                                        load_results = await asyncio.gather(*load_tasks)
                                        if not is_warmup:
                                            run_ctx_results.append(load_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in load_results):
                                            first_error = next(r.error for r in load_results if r.error)
                                            print(f"\n[Error] Stopping due to error in context load: {first_error}")
                                            raise BenchmarkFailure()

                                        # Phase 2: Inference
                                        print(f"  {run_label} (Inference, batch size {concurrency})...")
                                        inf_tasks = []
                                        for i in range(concurrency):
                                            context, prompt = prompt_batch[i]
                                            inf_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=prompt,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer
                                            ))

                                        batch_results = await asyncio.gather(*inf_tasks)
                                        if not is_warmup:
                                            run_std_results.append(batch_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                            first_error = next(r.error for r in batch_results if r.error)
                                            print(f"\n[Error] Stopping due to error in inference: {first_error}")
                                            raise BenchmarkFailure()

                                    else:
                                        # Standard Run
                                        print(f"  {run_label} (batch size {concurrency})...")
                                        expected_tokens = current_pp + current_depth
                                        batch_tasks = []
                                        for i in range(concurrency):
                                            context, prompt = prompt_batch[i]
                                            batch_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=prompt,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer
                                            ))

                                        batch_results = await asyncio.gather(*batch_tasks)
                                        if not is_warmup:
                                            run_std_results.append(batch_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                            first_error = next(r.error for r in batch_results if r.error)
                                            print(f"\n[Error] Stopping due to error in standard run: {first_error}")
                                            raise BenchmarkFailure()


                                    # Post Run Command
                                    if self.config.post_run_cmd:
                                        try:
                                            subprocess.run(self.config.post_run_cmd, shell=True, check=True)
                                        except subprocess.CalledProcessError as e:
                                            print(f"Post-run command failed: {e}")

                                    # Wait for idle between runs/configurations
                                    if self.config.idle_wait:
                                        await self.wait_for_idle(session)

                                # Aggregate and Record
                                if self.config.enable_prefix_caching and depth > 0:
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_ctx_results, latency, expected_ctx, is_context_phase=True, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)
                                else:
                                    # Standard run expected tokens = pp + depth (usually depth=0 or concatenated)
                                    # In the loop above: expected_tokens = current_pp + current_depth
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp + expected_ctx, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)

                self.results.metadata = BenchmarkMetadata(
                    version=__version__,
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                    latency_mode=self.config.latency_mode,
                    latency_ms=latency * 1000,
                    model=self.config.model,
                    prefix_caching_enabled=self.config.enable_prefix_caching,
                    max_concurrency=max(self.config.concurrency_levels) if self.config.concurrency_levels else 1
                )

            self.results.save_report(self.config.save_result, self.config.result_format, max(self.config.concurrency_levels) if self.config.concurrency_levels else 1)

        except (asyncio.CancelledError, KeyboardInterrupt, BenchmarkFailure) as e:
            if self.results.runs:
                should_save = True
                if isinstance(e, BenchmarkFailure) and self.config.no_results_on_fail:
                    should_save = False
                    print("\n[Failed] Results discarded per --no-results-on-fail.")

                if should_save:
                    print("\n[Interrupted/Failed] Saving partial results...")
                    if self.results.metadata is None:
                        self.results.metadata = BenchmarkMetadata(
                            version=__version__,
                            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                            latency_mode=self.config.latency_mode,
                            latency_ms=latency * 1000,
                            model=self.config.model,
                            prefix_caching_enabled=self.config.enable_prefix_caching,
                            max_concurrency=max_concurrency
                        )
                    self.results.save_report(self.config.save_result, self.config.result_format, max_concurrency)
            
            if isinstance(e, BenchmarkFailure):
                sys.exit(1)
            raise
