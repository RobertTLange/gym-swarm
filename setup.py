from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym_swarm',
      version='0.0.1',
      author='Robert Tjrako Lange',
      author_email='robert.t.lange@web.de',
      license='MIT',
      description="An OpenAI Gym Environment for Complex Swarm Behavior.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/RobertTLange/gym-swarm",
      install_requires=['numpy', 'scipy', 'sklearn', 'gym', 'matplotlib']
      )
