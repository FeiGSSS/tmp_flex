from setuptools import setup, find_packages
import os

# 读取 README.md 作为 long_description
try:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A multi-package setup for flexllmgen CPU and Eval Utils."

setup(
    name='my-llm-bench-project', # 给你的整个项目起个名字，这不是包名，而是安装的顶层项目名
    version='0.0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='Manages flexllmgen CPU and Eval Utils as separate packages.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/your-project',

    # !!! 关键部分 !!!
    # find_packages() 的作用是自动查找项目中的Python包。
    # 这里我们告诉它从当前目录（'.'）开始查找。
    # include 参数用于指定要包含的包名模式。
    # 
    # 'flexllmgen*' 会匹配 'flexllmgen' 包。
    # 'A1.A2.eval_utils' 会精确匹配到 'A1/A2/eval_utils' 这个路径下的 'eval_utils' 包。
    # 注意：如果你的包名就是 eval_utils，并且它在 A1/A2 目录下，
    # 那么 find_packages 自动找到的包名会是 'A1.A2.eval_utils'。
    # 如果你希望它直接以 'eval_utils' 导入，这需要额外处理。
    # 
    # 更直接的方法是使用 `package_dir` 来映射。
    # 如果要避免这种复杂性，通常会调整目录结构，让所有包都在顶层。

    packages=find_packages(
        where='.',  # 从当前目录开始查找
        include=['flexllmgen*', 'flexgen*', 'eval_utils*'] # 明确包含这两个路径作为包
    ),
    
    # 如果你想让 `eval_utils` 包在安装后直接以 `import eval_utils` 导入，
    # 而不是 `import A1.A2.eval_utils`，你可以使用 `package_dir`。
    # 但是，这会使你的项目结构有点不直观。
    # 
    # 例如：
    # package_dir={
    #     'flexllmgen': 'flexllmgen',
    #     'eval_utils': 'A1/A2/eval_utils',
    # },
    # packages=['flexllmgen', 'eval_utils'], # 显式列出你想安装的包名

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'transformers>=4.24.0',
        'tokenizers', 
        'nltk', 
        'numpy',
        'tqdm>=4.0.0',
        'evaluate', 
        'sqlitedict', 
        'jsonlines', 
        'zstandard', 
        'unitxt', 
        'rouge_score',
        'ipython', 
        'jupyter', 
        'matplotlib', 
        'modelscope', 
        'scipy', 
        'jinja2', 
        'pytest', 
        'wandb', 
        'pytablewriter',   
        'pulp',
        'attrs',
        'huggingface_hub', 
        'accelerate>=0.20.0',
        'packaging>=20.0',
        'sacrebleu', 
        'scikit-learn' 
    ],
)