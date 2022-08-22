DIR='../out/'

lindenmayer:
	python3 lindenmayer.py
	$(MAKE) convert

datagen:
	python3 datagen.py 1 1 1 10 10 3 5 True $(DIR)
	$(MAKE) convert

datagen_remote:
	./scripts/conda-activate.sh
	xvfb-run python3 datagen.py 100_000 1 5 5 10 5 6 False $(DIR)
	$(MAKE) convert

convert:
	@echo "Converting .ps files..."
	@./scripts/convert-ps.sh $(DIR)
	@echo "Removing leftover .ps files..."
	@rm $(DIR)/*.ps	

clean:
	@rm $(DIR)/*.png
	@rm $(DIR)/*grammar.txt

view:
	sxiv $(DIR)

grammars:
	./scripts/show_grammars.sh $(DIR)
