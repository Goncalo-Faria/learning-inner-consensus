BEGIN {FS="." };
	{ if( $1 ~ "model") {
		str = $2 ;
		sub(/ckpt-/, "", str);

		if( length(str) == 4)
			str = "0" str

		model[str] = 1
		} 
	}
END   { n = asorti(model,dest,cmp_str_val); for (i = 1; i <= n; i++) print dest[i] }