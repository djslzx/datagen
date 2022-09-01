ROOT := '../out/'
DIR := '../out/sep1'

lindenmayer:
	python3 -u lindenmayer.py $(DIR)
	$(MAKE) convert

datagen:
	$(MAKE) clean
	mkdir -p $(DIR)
	python3 -u datagen.py 5 1 6 1 4 5 10 3 5 $(DIR)
	$(MAKE) convert

datagen_remote:
	$(MAKE) clean
	./scripts/conda-activate.sh
	mkdir -p $(DIR)
	xvfb-run python3 -u datagen.py 100_000 1 5 2 5 5 15 5 6 $(DIR)

convert:
	@echo "Converting .ps files..."
	./scripts/convert-ps.sh $(DIR)
	@echo "Removing leftover .ps files..."
	@rm -f $(DIR)/*.ps	

clean:
	@rm -f $(DIR)/*.svg
	@rm -f $(DIR)/*.png
	@rm -f $(DIR)/*.ps
	@rm -f $(DIR)/*.dat
	@rm -f $(DIR)/*.log

view:
	sxiv $(DIR)

grammars:
	less $(DIR)/grammars.log
