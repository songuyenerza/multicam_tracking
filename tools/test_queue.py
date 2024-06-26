from queue import Queue
q = Queue(maxsize = 3)
print(q.qsize()) 
q.put('1')
q.put('2')
q.put('3')
print("\nFull: ", q.full()) 
print("\nElements dequeued from the queue")
print(q.get())
print(q.get())
print("size of queue: ", q.qsize())

print(q.get())
print("size of queue: ", q.qsize())
print("\nEmpty: ", q.empty())
q.put(1)
q.put(2)
q.put(3)
if q.qsize() == 3:
    q.get()
    q.put(4)

print(q.get())
print(q.get())
print(q.get())

print("size of queue: ", q.qsize())

print("\nEmpty: ", q.empty()) 
print("Full: ", q.full())