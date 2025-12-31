#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for evolux.evolve.planner
"""
import asyncio
import unittest

from evolux.evolve import EvolveAgent
from evolux.evolve.context import load_config


class TestEvolveAgent(unittest.TestCase):
    def test_evolve_agent(self):
        asyncio.run(self._test_evolve_agent())

    async def _test_evolve_agent(self):
        config = load_config("./tests/evolux/evolve/evolve_agent/config.yaml")
        evolve_agent = EvolveAgent(config)

        result_msg = await evolve_agent.run()
        print(result_msg)


if __name__ == "__main__":
    unittest.main()
