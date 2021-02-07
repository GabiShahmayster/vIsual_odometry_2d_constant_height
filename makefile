venv:
	@echo Setting up Python virtual environment
	python3 -m venv venv_python
	#source test_env_indoor_robotics/bin/activate
	venv_python/bin/pip3 install -r requirements.txt
