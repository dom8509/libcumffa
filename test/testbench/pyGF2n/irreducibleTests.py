# f: the polynomial to test for irreducibility
# p: the distinct prime divisors of n = degree(f)
# 
# returns: 0 = irreducible
# 		   1 = reducible
def rabin1( f, p ):
	d = degree(f)

	n = [d/pi for pi in range(1, len(p))]

	for i in range(1, len(n)):
		g = gcd(f, (pow(pow(x, 2), n[i]) - x) % f)
		if g != 1: return 1 #reducible
		                    
	g = (pow(pow(x, 2), d) - x) % f
	if g != 1: 
		return 1
	else:
		return 0

def rabin2( f, p ):
	d = degree(f)
	n = 0
	h = x

	n = [d/pi for pi in range(1, len(p))]

	sort(n) # n[0] < n[1] < ... < n[len(p)]

	for i in range(1, len(n)):
		h
		g = gcd(f, (pow(pow(x, 2), n[i]) - x) % f)
		if g != 1: return 1 #reducible
		                    
	g = (pow(pow(x, 2), d) - x) % f
	if g != 1: 
		return 1
	else:
		return 0

def ben_or():
	pass