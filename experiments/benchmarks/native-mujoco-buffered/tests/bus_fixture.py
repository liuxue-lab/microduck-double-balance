"""In-memory ROS endpoint fixture; transport still uses real sockets."""
import queue
import threading
from types import SimpleNamespace

class FakeBus:
    def __init__(self):
        self.guard = threading.Lock()
        self.publishers, self.subscribers = {}, {}

    def endpoints(self, args):
        bus = self
        class Message:
            def __init__(self, data):
                self.data = data
        class Node:
            def __init__(self):
                self.queue = queue.Queue()
                self.owned_p, self.owned_s = [], []
            def create_publisher(self, kind, topic, qos):
                with bus.guard:
                    bus.publishers.setdefault(topic, []).append(self)
                self.owned_p.append(topic)
                def publish(message):
                    with bus.guard:
                        targets = list(bus.subscribers.get(topic, []))
                    for node, callback in targets:
                        node.queue.put((callback, Message(message.data)))
                return SimpleNamespace(publish=publish)
            def create_subscription(self, kind, topic, callback, qos):
                with bus.guard:
                    bus.subscribers.setdefault(topic, []).append((self, callback))
                self.owned_s.append(topic)
                return object()
            def count_publishers(self, topic):
                with bus.guard:
                    return len(bus.publishers.get(topic, []))
            def count_subscribers(self, topic):
                with bus.guard:
                    return len(bus.subscribers.get(topic, []))
            def destroy_node(self):
                with bus.guard:
                    for topic in self.owned_p:
                        bus.publishers[topic].remove(self)
                    for topic in self.owned_s:
                        bus.subscribers[topic] = [(n, c) for n, c in bus.subscribers[topic] if n is not self]
        def spin_once(node, timeout_sec):
            try:
                callback, message = node.queue.get(timeout=timeout_sec)
            except queue.Empty:
                return
            callback(message)
        ros = SimpleNamespace(spin_once=spin_once, shutdown=lambda: None)
        return ros, Node(), {}, Message, '/fixture/' + args.session

