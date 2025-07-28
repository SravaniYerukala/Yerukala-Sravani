class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def reverse_iterative(self):
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev
    def push(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def print_list(self):
        temp = self.head
        while temp:
            print(temp.data, end=" ")
            temp = temp.next
        print("None")
llist = LinkedList()
llist.push(4)
llist.push(3)
llist.push(2)
llist.push(1)
print("Original Linked List:")
llist.print_list()
llist.reverse_iterative()
print("Reversed Linked List:")
llist.print_list()

            

