import itertools
import sys

def make_njklists(Lmax, Nmodes):
    njklists = list()
    for njklist in itertools.product(range(Lmax+1), repeat=Nmodes):
        if sum(njklist) <= Lmax:
            njklists.append(njklist)
    return njklists

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print "usage: %s max length"%(sys.argv[0])
        sys.exit()

    max = int(args[0])
    length = int(args[1])

    print 'def product(max, length):'
    for m in range(1,max+1):
        print '    if max == %d:'%(m)
        for l in range(1,length+1):
            print '        if length == %d:'%(l)
            print '            return', make_njklists(m,l)
    print '    raise NotImplementedError'

if __name__ == '__main__':
    main()
