CXX := gawk
FLAGS= -f

exec:	
	ls $(DIR) > $(DIR)/ls.txt
	$(CXX) $(FLAGS) extract.gawk $(DIR)/ls.txt > tmp.txt
	mv tmp.txt $(DIR)/ls.txt
