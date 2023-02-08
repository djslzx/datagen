import heapq

class MaxHeap():
    """
    Class wrapper over Python heap implementation
    Implements a max heap
    """
    def __init__(self):
        self.heap = []
        self.counter = 0 # counter used to break ties

    def push(self, priority, item):
        """
        Pushes a new element to the heap with a given priority
        """
        self.counter += 1
        heapq.heappush(self.heap, (-priority, self.counter, item))

    def pop(self):
        """
        Pops and returns the element from the heap with the highest priority
        """
        assert not self.empty(), "cannot pop from empty heap"
        
        priority, _, item = heapq.heappop(self.heap)
        return -priority, item

    def peek(self):
        """
        Returns the highest priority element from the heap without removing it from the heap
        """
        assert not self.empty(), "cannot peak from empty heap"        
        priority, _, item = self.heap[0]
        return -priority, item

    def empty(self):
        """
        Returns whether the heap is empty
        """
        return len(self.heap) == 0
