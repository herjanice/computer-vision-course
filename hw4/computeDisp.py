import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    window_size = 5
    center = window_size // 2
    row = []
    col = []

    Il_pad = np.pad(Il, [(center, center), (center, center), (0, 0)], mode='edge')
    Ir_pad = np.pad(Ir, [(center, center), (center, center), (0, 0)], mode='edge')

    for i in range(window_size):
        for j in range(window_size):
            if i != center or j!= center:
                row.append(i)
                col.append(j)

    Il_binary =  np.zeros((h,w,ch,len(row)), dtype=np.bool)
    Ir_binary =  np.zeros((h,w,ch,len(row)), dtype=np.bool)
    for i in range(len(row)):
        Il_binary[:,:,:,i] = Il < Il_pad[row[i]:row[i]+h, col[i]:col[i]+w, :]
        Ir_binary[:,:,:,i] = Ir < Ir_pad[row[i]:row[i]+h, col[i]:col[i]+w, :]

    # Calculate cost for each disparity
    Il_cost =  np.zeros((max_disp+1,h,w,ch), dtype=np.float32)
    Ir_cost =  np.zeros((max_disp+1,h,w,ch), dtype=np.float32)
    for d in range(max_disp+1):
        cost_l = np.logical_xor(Il_binary[:,d:,:,:], Ir_binary[:,:w-d,:,:])
        cost_l = np.sum(cost_l, axis = 3)
        cost_l = np.pad(cost_l, [(0, 0), (d, 0), (0, 0)], mode='edge')

        Il_cost[d] = cost_l

        cost_r = np.logical_xor(Il_binary[:,d:,:,:], Ir_binary[:,:w-d,:,:])
        cost_r = np.sum(cost_r, axis = 3)
        cost_r = np.pad(cost_r, [(0, 0), (0, d), (0, 0)], mode='edge')

        Ir_cost[d] = cost_r 

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    for d in range(max_disp+1):
        Il_cost[d] = xip.jointBilateralFilter(Il, Il_cost[d], 30, 5, 5)
        Ir_cost[d] = xip.jointBilateralFilter(Ir, Ir_cost[d], 30, 5, 5)
        # Il_cost[d] = xip.guidedFilter(Il, Il_cost[d], 5, 75)
        # Ir_cost[d] = xip.guidedFilter(Ir, Ir_cost[d], 5, 75)
    Il_cost = np.sum(Il_cost, axis=-1)
    Ir_cost = np.sum(Ir_cost, axis=-1)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    Il_map = np.argmin(Il_cost, axis=0)
    Ir_map = np.argmin(Ir_cost, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    Il_enhance = np.zeros((h,w))
    Ir_enhance = np.zeros((h,w))
    for i in range(h):
        closest = None
        for j in range(w):
            if Ir_map[i, j-Il_map[i,j]] == Il_map[i,j]:
                closest = Il_map[i,j]
                Il_enhance[i,j] = Il_map[i,j]
            else:
                if closest == None:
                    Il_enhance[i,j] = np.Inf
                else:
                    Il_enhance[i,j] = closest
    
    w_inv = np.arange(w-1,-1,-1)
    for i in range(h):
        closest = None
        for j in w_inv:
            if Ir_map[i, j-Il_map[i,j]] == Il_map[i,j]:
                closest = Il_map[i,j]
                Ir_enhance[i,j] = Il_map[i,j]
            else:
                if closest == None:
                    Ir_enhance[i,j] = np.Inf
                else:
                    Ir_enhance[i,j] = closest
    
    Il_disparity_map = np.minimum(Il_enhance, Ir_enhance).astype(np.uint8)

    Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    Il_gray = cv2.medianBlur(Il_gray, 3)
    labels = xip.weightedMedianFilter(Il_gray, Il_disparity_map, r=11, sigma=0.001)
    labels = cv2.medianBlur(labels, 7)


    return labels.astype(np.uint8)
    