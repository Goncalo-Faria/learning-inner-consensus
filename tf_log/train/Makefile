CXX := gawk
FLAGS= -f

exec:	
	ls $(DIR) > $(DIR)/ls.txt
	$(CXX) $(FLAGS) $(DIRWK)/extract.gawk $(DIR)/ls.txt > tmp.txt
	mv tmp.txt $(DIR)/ls.txt
