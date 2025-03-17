# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
从文档生成问题的LangChain实现

这个模块基于LangChain实现了从文本文档生成问题列表的功能。
它使用VLLM服务器作为后端，通过LangChain的链式操作和结构化输出解析，
以并行批处理的方式高效地处理大规模数据集。

用法示例:
    python promble_generate_langchain.py --dataset /path/to/dataset --output /path/to/output --limit 100
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import datasets
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("problem_generation.log"), logging.StreamHandler()],
)
logger = logging.getLogger("problem_generator")

# 默认配置
model = "/test5/.modlescope_chace/models/Qwen/Qwen2.5-14B-Instruct-AWQ"
vllm_server_url = "http://localhost:8000/v1"
prompt_template = """你是一位从文档中提取信息以出题的专家用户。你会收到从文档中提取的一页内容，然后根据仅提供的这段文字，编写一份无需的问题列表, 总共返回1~2个你觉得最值得出问题, 要求问题能够覆盖到文档中的所有知识点。

<问题列表示例>
{{
    "problems": [
        "问题1",
        "问题2",
    ]
}}
</问题列表示例>

<文档内容>
{text}
</文档内容>

现在请你返回问题列表，注意只需要返回问题列表，不要返回其他内容。
"""
prompt_column = "text"
temperature = 0.7
top_p = 0.9
max_new_tokens = 8192
input_batch_size = 64
num_threads = 4
timeout = 900
retries = 3
dataset_path = "/test5/myh_dev/Aquila_com/train_5.2_1562"
output_path = "/test5/myh_dev/Aquila_com/train_5.2_1562_problem"


class Problem(BaseModel):
    problems: List[str] = Field(..., description="问题列表")


def get_problem_generator(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_template_str: str = "{text}",
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.9,
    timeout: int = 900,
    retries: int = 3,
) -> Runnable:
    """创建问题生成器链

    构建一个能够从文本生成问题列表的LangChain链，包括提示模板、LLM配置和输出解析器。

    参数:
        model: 模型名称或路径
        base_url: VLLM服务器的URL
        prompt_column: 输入数据集中的文本列名
        prompt_template_str: 提示模板字符串
        temperature: 生成多样性参数
        top_p: 核采样参数
        timeout: 请求超时时间(秒)
        retries: 最大重试次数

    返回:
        配置好的LangChain链对象
    """

    # 创建提示模板
    prompt = PromptTemplate(
        template=prompt_template_str,
    )

    llm = ChatOpenAI(
        base_url=base_url,
        api_key="something",  # 占位符
        model_name=model,
        temperature=temperature if temperature is not None else 0.7,
        top_p=top_p if top_p is not None else 0.9,
        request_timeout=timeout,
        max_retries=retries,
        streaming=False,  # 批处理模式下关闭流式输出
    )
    parser = PydanticOutputParser(pydantic_object=Problem)
    # 创建LangChain链
    chain = prompt | llm | parser
    return chain.with_retry(
        retry_if_exception_type=(ValueError,),
        wait_exponential_jitter=True,
        stop_after_attempt=retries,
    )


async def process_dataset(
    dataset: datasets.Dataset,
    chain: Runnable,
    prompt_column: str,
    max_concurrent: int = 100,  # 最大并发数
) -> List[Dict]:
    """处理数据集，生成问题列表

    Args:
        dataset: 输入数据集
        chain: LangChain链
        prompt_column: 文本列名
        max_concurrent: 最大并发请求数
    """
    results = []
    # 使用信号量控制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    # 用于存储所有任务
    tasks = []
    # 用于存储结果的队列
    result_queue = asyncio.Queue()

    async def process_single_item(item):
        async with semaphore:  # 使用信号量控制并发
            text = item[prompt_column]
            try:
                result = await chain.ainvoke({"text": text})
                problems = result.problems
                # 将结果放入队列
                for problem in problems:
                    await result_queue.put({
                        "context": text,
                        "problem": problem,
                    })
            except Exception as e:
                logger.error(f"处理样本时发生错误: {e}")

    async def collect_results():
        """从队列中收集结果"""
        processed_count = 0
        while True:
            try:
                result = await result_queue.get()
                results.append(result)
                processed_count += 1

                if processed_count % max_concurrent == 0:
                    logger.info(f"已处理 {processed_count} 个结果")

                result_queue.task_done()
            except asyncio.CancelledError:
                break

    # 启动结果收集器
    collector = asyncio.create_task(collect_results())

    # 创建所有处理任务
    for item in dataset:
        task = asyncio.create_task(process_single_item(item))
        tasks.append(task)

    # 等待所有任务完成
    await asyncio.gather(*tasks)
    # 等待队列处理完成
    await result_queue.join()
    # 取消结果收集器
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    return results


async def main():
    """主函数，解析命令行参数并执行生成流程"""
    import argparse

    import datasets

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="使用LangChain从文档生成问题")
    parser.add_argument("--model", type=str, default=model, help="模型路径")
    parser.add_argument(
        "--server-url", type=str, default=vllm_server_url, help="VLLM服务器URL"
    )
    parser.add_argument(
        "--dataset", type=str, default=dataset_path, help="输入数据集路径"
    )
    parser.add_argument("--output", type=str, default=output_path, help="输出路径")
    parser.add_argument(
        "--batch-size", type=int, default=input_batch_size, help="批处理大小"
    )
    parser.add_argument("--threads", type=int, default=num_threads, help="线程数")
    parser.add_argument(
        "--temperature", type=float, default=temperature, help="生成温度"
    )
    parser.add_argument("--top-p", type=float, default=top_p, help="核采样参数")
    parser.add_argument(
        "--max-tokens", type=int, default=max_new_tokens, help="最大生成token数"
    )
    parser.add_argument("--timeout", type=int, default=timeout, help="请求超时时间(秒)")
    parser.add_argument("--retries", type=int, default=retries, help="最大重试次数")
    parser.add_argument(
        "--limit", type=int, default=-1, help="处理样本数量限制，-1表示全部"
    )
    parser.add_argument(
        "--save-intermediate", action="store_true", help="是否保存中间结果"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="json,hf",
        help="输出格式，逗号分隔，支持json和hf",
    )
    args = parser.parse_args()

    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = datasets.load_from_disk(args.dataset)

    # 应用限制(如果有)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    logger.info(f"已加载 {len(dataset)} 个样本")

    # 创建问题生成器
    logger.info("初始化问题生成链...")
    problem_chain = get_problem_generator(
        model=args.model,
        base_url=args.server_url,
        prompt_template_str=prompt_template,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        retries=args.retries,
    )

    # 处理数据集
    logger.info("开始生成问题...")
    start_time = time.time()
    results: List[Dict] = await process_dataset(
        dataset=dataset,
        chain=problem_chain,
        prompt_column=prompt_column,
    )
    end_time = time.time()
    logger.info(f"问题生成完成! 总耗时: {end_time - start_time:.2f}秒")

    # 保存结果
    logger.info("保存结果...")
    results_dataset = datasets.Dataset.from_list(results)
    results_dataset.save_to_disk(args.output)
    logger.info(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
