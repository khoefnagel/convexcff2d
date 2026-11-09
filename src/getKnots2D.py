"""
getKnots2D.py

Contains getKnots2D function to find the intersection knots between 
a facet and the subgrid lines in 2D.

requires: numpy, matplotlib (only for testMode)
"""

import numpy as np

def getKnots2D(x1 : float, y1 : float, x2 : float, y2 : float, testMode : bool = False):
    """Find the intersection knots between a facet and the subgrid lines in 2D, defined by:
        y=k/2,
        x=k/2,
        y-x=k,
        y+x=k,
    where k is an integer. 
    
    Parameters
    ----------
    x1 : float
        x-coordinate of the first vertex of the facet (line segment)
    y1 : float
        y-coordinate of the first vertex of the facet (line segment)
    x2 : float
        x-coordinate of the second vertex of the facet (line segment)
    y2 : float
        y-coordinate of the second vertex of the facet (line segment)
    testMode : bool, optional
        Whether to plot the facet and its knots
        (default is False)

    Returns
    -------
    knotsSorted : array_like
        Array of shape (Nk, 2) with Nk the number of knots. The first column
        corresponds to x-coordinates, the second to y-coordinates. The knots
        are sorted by increasing distance to the first vertex."""

    # To prevent division by zero when x1 and x2 are equal, the x and y are switched
    # in this case. They are swapped back correctly when returned.
    switchXY = False
    if np.linalg.norm(x2-x1) < 1e-12:
        y1, y2, x1, x2 = x1, x2, y1, y2
        switchXY = True

    # Find line coefficients of the facet as a line segment (y=a*x+b)
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    #---------------------------------------------------------
    # Intersection knots with vertical lines
    verKnotsX = np.arange(np.ceil(2*min(x1, x2)).astype(int),
                         np.ceil(2*max(x1, x2)).astype(int))/2
    verKnotsY = a*verKnotsX+b

    #---------------------------------------------------------
    # Intersection knots with horizontal lines
    horKnotsY = np.arange(np.ceil(2*min(y1, y2)).astype(int),
                         np.ceil(2*max(y1, y2)).astype(int))/2
    horKnotsX = (horKnotsY-b)/a

    #---------------------------------------------------------
    # Intersection knots with diagonal lines with slope +1
    diagUpk1 = (a-1)*x1+b
    diagUpk2 = (a-1)*x2+b
    diagUpK = np.arange(np.ceil(min(diagUpk1, diagUpk2)).astype(int),
                        np.ceil(max(diagUpk1, diagUpk2)).astype(int))
    diagUpX = (diagUpK-b)/(a-1)
    diagUpY = diagUpK + diagUpX

    #---------------------------------------------------------
    # Intersection knots with diagonal lines with slope -1
    diagDownk1 = (a+1)*x1 + b
    diagDownk2 = (a+1)*x2 + b
    diagDownK = np.arange(np.ceil(min(diagDownk1, diagDownk2)).astype(int),
                          np.ceil(max(diagDownk1, diagDownk2)).astype(int))
    diagDownX = (diagDownK-b)/(a+1)
    diagDownY = diagDownK - diagDownX

    #---------------------------------------------------------
    # Collect all intersection knots in a (2, Nk) array
    knots = np.vstack((np.concatenate([np.array([x1]), verKnotsX, horKnotsX,
                                       diagUpX, diagDownX, np.array([x2])]),
                       np.concatenate([np.array([y1]), verKnotsY, horKnotsY,
                                       diagUpY, diagDownY, np.array([y2])])))

    # Distance to first vertex (in flipped coordinates if switchXY == True)
    dist = np.linalg.norm(knots - np.array([[x1], [y1]]), axis=0)

    # Sort knots by distance from starting vertex
    sortInds = np.argsort(dist)
    knotsSorted = knots[:,sortInds]

    # Switch back coordinates if needed
    if switchXY:
        knotsSorted = np.vstack((knotsSorted[1], knotsSorted[0]))
 
    # Return knots if not in testmode (default)
    if not testMode:
        return knotsSorted.T

    # The next plotting code is only executed if testmode is on

    # Switch variables also correctly for plotting
    if switchXY:
        verKnotsX, verKnotsY = verKnotsY, verKnotsX
        horKnotsX, horKnotsY = horKnotsY, horKnotsX
        diagUpX, diagUpY = diagUpY, diagUpX
        diagDownX, diagDownY = diagDownY, diagDownX
        y1, y2, x1, x2 = x1, x2, y1, y2
        

    # Plot the facet
    plt.plot([x1, x2], [y1, y2], c='g', linestyle='dashed')

    # Plot the vertical subgrid lines
    plt.scatter(verKnotsX, verKnotsY, marker='x', c='r', zorder=100,
                label='Vertical knot')
    for x in range(2*np.floor(min(x1,x2)).astype(int),
                   2*np.ceil(max(x1,x2)).astype(int), 1):
        plt.axvline(x/2, c='k')

    # Plot the horizontal subgrid lines
    plt.scatter(horKnotsX, horKnotsY, marker='o', c='purple', zorder=100,
                label='Horizontal knot')
    for y in range(2*np.floor(min(y1,y2)).astype(int),
                   2*np.ceil(max(y1,y2)).astype(int), 1):
        plt.axhline(y/2, c='k')

    # Plot the diagonal up subgrid lines
    plt.scatter(diagUpX, diagUpY, marker='*', c='b', zorder=100,
                label='Diagonal up knot')
    for k in diagUpK:
        plt.axline((0, k), slope=1, c='k')

    # Plot the diagonal down subgrid lines
    plt.scatter(diagDownX, diagDownY, marker='d', c='darkorange', zorder=100,
                label='Diagonal down knot')
    for k in diagDownK:
        plt.axline((0, k), slope=-1, c='k')

    #Plot decoration
    plt.xlim(min(x1,x2)-0.1, max(x1,x2)+0.1)
    plt.ylim(min(y1,y2)-0.1, max(y1,y2)+0.1)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(ncol=2, bbox_to_anchor=(0.05,1.05))
    plt.tight_layout()
    plt.show()


#Execute in testmode with a test facet if this script is run directly
if __name__ == '__main__':
    
    x1, y1 = -1.3, 1.4
    x2, y2 = 1.1, 0.1

    import matplotlib.pyplot as plt

    print(getKnots2D(x1, y1, x2, y2, testMode=False))
    getKnots2D(x1, y1, x2, y2, testMode=True)
