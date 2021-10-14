#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from habitat.core.benchmark import Benchmark
from habitat.core.logging import logger


class Challenge(Benchmark):
    def __init__(self, phase = None):
        config_path = "configs/tasks/multinav_mp3d.yaml"
        super().__init__(config_path, phase = phase)
        self.num_episodes = None
    def submit(self, agent):
        metrics = super().evaluate(agent, num_episodes = self.num_episodes)
        print("Progress:", metrics["percentage_success"])
        print("PPL:", metrics["pspl"])
        print("Success:", metrics["success"])
        print("SPL:", metrics["mspl"])
