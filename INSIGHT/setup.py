from setuptools import setup, find_packages

setup(
    name="insight-viz",
    version="0.1.0",
    description="Genome data visualization toolkit",
    author="douymLab",
    packages=find_packages(),
    install_requires=[
        "duckdb",
        "pysam",
        "requests",
        "numpy",
        "scipy",
        "Pillow",
        "cairosvg",
        "pydantic",
        "psutil",
        "tqdm",
        "jinja2",
        "PyYAML",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "bayesmonstr2insights=in_sight.commands.bayesmonstr2insights:main",
            "cf2insights=in_sight.commands.cf2insights:main",
        ],
    },
    package_data={
        "in_sight": ["r_script/*.R", "templates/*"],
    },
    include_package_data=True,
    python_requires=">=3.8",
)
