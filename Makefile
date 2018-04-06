all: release force_look

release: tests force_look

lib: force_look
	@$(MAKE) -C src/lib

tests: force_look
	@echo "Compiling library ..."
	@$(MAKE) -C src/lib
	
	@echo "Compiling testbench ..."
	@rm -Rf test/testbench/c_wrapper/build
	@$(MAKE) -C test/testbench/c_wrapper

	@cd test/unit/ && python runUnitTests.py -gTestMyMul && cd ../..

	@echo "Compiling metrics wrapper ..."
	@$(MAKE) -C test/testbench/metrics_wrapper clean
	@$(MAKE) -C test/testbench/metrics_wrapper
	cp test/testbench/metrics_wrapper/runFunction test/performance/
	cp test/testbench/metrics_wrapper/runFunction2047 test/performance/	

check: force_look
	@echo "Compiling library ..."
	@$(MAKE) -C src/lib
	
	@echo "Compiling testbench ..."
	@rm -Rf test/testbench/c_wrapper/build
	@$(MAKE) -C test/testbench/c_wrapper
	@echo "Running unit tests ..."
	@cd test/unit/ && python runUnitTests.py && cd ../..

	@echo "Compiling testcode ..."
	@$(MAKE) test -C src/lib
	@echo "Running testcode ..."
	./src/lib/build/test

	@echo "All checks passed!"

force_look: 
	@true
