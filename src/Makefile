DIR := '../out/aug25'

lindenmayer:
	python3 -u lindenmayer.py
	$(MAKE) convert

datagen:
	$(MAKE) clean
	mkdir -p $(DIR)
	python3 -u datagen.py 4 1 4 2 4 2 6 3 5 $(DIR)
	# $(MAKE) convert

datagen_remote:
	$(MAKE) clean
	./scripts/conda-activate.sh
	mkdir -p $(DIR)
	xvfb-run python3 -u datagen.py 100_000 1 5 2 5 5 15 5 6 $(DIR)
	# $(MAKE) convert

convert:
	@echo "Converting .ps files..."
	./scripts/convert-ps.sh $(DIR)
	@echo "Removing leftover .ps files..."
	@rm $(DIR)/*.ps	

clean:
	@rm -f $(DIR)/*.svg
	@rm -f $(DIR)/*.dat
	@rm -f $(DIR)/*.log
	@rm -f $(DIR)/*/*.svg
	@rm -f $(DIR)/*/*.dat
	@rm -f $(DIR)/*/*.log

view:
	sxiv $(DIR)

grammars:
	less $(DIR)/grammars.log
