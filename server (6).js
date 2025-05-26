const express = require('express');
const app = express();
const PORT = 3000;
const cors = require('cors');
app.use(cors());
// Example GET route: /get-letter?filename=example.jpg
app.get('/get-letter', (req, res) => {
    const filename = req.query.filename;

    if (!filename) {
        return res.status(400).json({ error: 'Filename is required as a query parameter' });
    }
    var index=3;
    const f = filename.charAt(0).toUpperCase();
    if(f==='C'){
    index=0;
    }else if(f==='R'){
    index = 3;
    }else if(f==='N'){
    index= 2;
    }else{
    index = 1;
    }
    res.json({
        prediction: index
    });
});

const prg3 = `
import heapq

def astar(start, goal, neighbors, h):
    class Node:
        def __init__(self, s, p=None, g=0): self.s, self.p, self.g = s, p, g
        def f(self): return self.g + h(self.s)

    open_set = [(h(start), id(start), Node(start))]
    closed = set()

    while open_set:
        _, _, curr = heapq.heappop(open_set)
        if curr.s == goal:
            path = []
            while curr: path.append(curr.s); curr = curr.p
            return path[::-1]
        closed.add(curr.s)
        for ns in neighbors(curr.s):
            if ns in closed: continue
            n = Node(ns, curr, curr.g + 1)
            if any(ns == x.s for _, _, x in open_set): continue
            heapq.heappush(open_set, (n.f(), id(n), n))
    return None

neighbors = lambda s: [(s[0]+dx, s[1]+dy) for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]]
heuristic = lambda s: abs(4 - s[0]) + abs(4 - s[1])

path = astar((0, 0), (4, 4), neighbors, heuristic)
print("Path:", path)
`;
app.get('/prg3', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(prg3);   // return the Python code
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
module.exports = app;
