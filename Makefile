# GraphForge 开发任务（作者: 晨星）
PY ?= python

.PHONY: help install lint test demo tier1 clean docker

help:
	@echo "install   安装依赖"
	@echo "lint      ruff 静态检查"
	@echo "test      pytest 单测"
	@echo "demo      端到端演示（落盘 benchmark.json）"
	@echo "tier1     离线兜底路径冒烟"
	@echo "docker    构建容器镜像"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install ruff pytest

lint:
	$(PY) -m ruff check graphforge tests examples

test:
	$(PY) -m pytest -q

demo:
	$(PY) examples/run_demo.py --seeds 0 1 2

tier1:
	$(PY) -m graphforge.cli run --graph karate --tier1-only

docker:
	docker build -t graphforge:0.1.0 .

clean:
	@rm -rf .pytest_cache .ruff_cache build dist
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
