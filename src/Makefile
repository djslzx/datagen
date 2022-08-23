DIR := '../out'

lindenmayer:
	python3 lindenmayer.py
	$(MAKE) convert

datagen:
	python3 datagen.py 1 2 4 10 10 3 5 $(DIR)
	$(MAKE) convert

datagen_remote:
	./scripts/conda-activate.sh
	xvfb-run python3 datagen.py 100_000 2 5 5 15 5 6 $(DIR)
	$(MAKE) convert

convert:
	@echo "Converting .ps files..."
	./scripts/convert-ps.sh $(DIR)
	@echo "Removing leftover .ps files..."
	@rm $(DIR)/*.ps	

clean:
	@rm $(DIR)/*.grammar
	@rm $(DIR)/*.png

view:
	sxiv $(DIR)

grammars:
	./scripts/show_grammars.sh $(DIR)
