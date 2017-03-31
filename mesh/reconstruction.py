import mesh.patch as patch
import mesh.reconstruction_f as reconstruction_f
import mesh.array_indexer as ai

def limit(data, myg, idir, limiter):

    if limiter < 10:
        if limiter == 0:
            limit_func = reconstruction_f.nolimit
        elif limiter == 1:
            limit_func = reconstruction_f.limit2
        else:
            limit_func = reconstruction_f.limit4

        return ai.ArrayIndexer(d=limit_func(idir, data, myg.qx, myg.qy, myg.ng), 
                                   grid=myg)
    else:
        ldax, lday = reconstruction_f.multid_limit(a, qx, qy, myg.ng)
        return ai.ArrayIndexer(d=ldax, grid=myg), ai.ArrayIndexer(d=lday, grid=myg)
