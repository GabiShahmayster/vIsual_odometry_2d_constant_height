venv:
	@echo Setting up Python virtual environment
	python3 -m venv test_env_indoor_robotics
	#source test_env_indoor_robotics/bin/activate
	test_env_indoor_robotics/bin/pip3 install -r requirements.txt
