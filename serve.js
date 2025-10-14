const express = require('express');
const app = express();
const PORT = 3000;
const cors = require('cors');
app.use(cors());
// Example GET route: /get-letter?filename=example.jpg
const info1 = `
set ns [new Simulator]

set nf [open prog1.nam w]
$ns namtrace-all $nf
set nd [open prog1.tr w]
$ns trace-all $nd

proc finish { } {
global ns nf nd
$ns flush-trace
close $nf
close $nd
exec nam prog1.nam &
exit 0
}

set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]

$ns duplex-link $n0 $n2 200Mb 10ms DropTail
$ns duplex-link $n1 $n2 100Mb 5ms DropTail
$ns duplex-link $n2 $n3 1Mb 1000ms DropTail

$ns queue-limit $n0 $n2 10
$ns queue-limit $n1 $n2 10

set udp0 [new Agent/UDP]
$ns attach-agent $n0 $udp0
set cbr0 [new Application/Traffic/CBR]
$cbr0 set packetSize_ 500
$cbr0 set interval_ 0.005
$cbr0 attach-agent $udp0

set udp1 [new Agent/UDP]
$ns attach-agent $n1 $udp1
set cbr1 [new Application/Traffic/CBR]
$cbr1 set packetSize_ 500
$cbr1 set interval_ 0.005
$cbr1 attach-agent $udp1

set null0 [new Agent/Null]
$ns attach-agent $n3 $null0

$ns connect $udp0 $null0
$ns connect $udp1 $null0

$ns at 0.1 "$cbr0 start"
$ns at 0.2 "$cbr1 start"
$ns at 1.0 "finish"

$ns run
`;
const info2 = `
set ns [new Simulator]

set nf [open prog2.nam w]
$ns namtrace-all $nf
set nd [open prog2.tr w]
$ns trace-all $nd

proc finish {} {
global ns nf nd
$ns flush-trace
close $nf
close $nd
exec nam prog2.nam &
exit 0
}

Agent/Ping instproc recv {from rtt} {
$self instvar node_
puts "Node [$node_ id] received ping reply from $from with RTT = $rtt ms"
}

set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]
set n6 [$ns node]

$ns duplex-link $n1 $n0 1Mb 10ms DropTail
$ns duplex-link $n2 $n0 1Mb 10ms DropTail
$ns duplex-link $n3 $n0 1Mb 10ms DropTail
$ns duplex-link $n4 $n0 50Kb 10ms DropTail
$ns duplex-link $n5 $n0 50Kb 10ms DropTail
$ns duplex-link $n6 $n0 50Kb 10ms DropTail

$ns queue-limit $n0 $n4 3
$ns queue-limit $n0 $n5 2
$ns queue-limit $n0 $n6 2

set p1 [new Agent/Ping]
set p2 [new Agent/Ping]
set p3 [new Agent/Ping]
set p4 [new Agent/Ping]
set p5 [new Agent/Ping]
set p6 [new Agent/Ping]

$ns attach-agent $n1 $p1
$ns attach-agent $n2 $p2
$ns attach-agent $n3 $p3
$ns attach-agent $n4 $p4
$ns attach-agent $n5 $p5
$ns attach-agent $n6 $p6

$ns connect $p1 $p4
$ns connect $p2 $p5
$ns connect $p3 $p6

for {set i 0} {$i < 200} {incr i} {
    $ns at [expr 0.1 + $i*0.01] "$p1 send"
    $ns at [expr 0.1 + $i*0.01] "$p2 send"
    $ns at [expr 0.1 + $i*0.01] "$p3 send"
}

$ns at 5.0 "finish"

$ns run

`;
const info3 = `

set ns [new Simulator]

set nf [open prog3.nam w]
$ns namtrace-all $nf

set nd [open prog3.tr w]
$ns trace-all $nd

$ns color 1 Blue
$ns color 2 Red

proc finish { } {
    global ns nf nd
    $ns flush-trace
    close $nf
    close $nd
    exec nam prog3.nam &
    exit 0
}

set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]
set n6 [$ns node]
set n7 [$ns node]
set n8 [$ns node]

$n7 shape box
$n7 color Blue
$n8 shape hexagon
$n8 color Red

$ns duplex-link $n1 $n0 2Mb 10ms DropTail
$ns duplex-link $n2 $n0 2Mb 10ms DropTail
$ns duplex-link $n0 $n3 1Mb 20ms DropTail
$ns make-lan "$n3 $n4 $n5 $n6 $n7 $n8" 512Kb 40ms LL Queue/DropTail Mac/802_3

$ns duplex-link-op $n1 $n0 orient right-down
$ns duplex-link-op $n2 $n0 orient right-up
$ns duplex-link-op $n0 $n3 orient right

$ns queue-limit $n0 $n3 20

set tcp1 [new Agent/TCP/Vegas]
$ns attach-agent $n1 $tcp1
set sink1 [new Agent/TCPSink]
$ns attach-agent $n7 $sink1
$ns connect $tcp1 $sink1
$tcp1 set class_ 1
$tcp1 set packetSize_ 55
set ftp1 [new Application/FTP]
$ftp1 attach-agent $tcp1

set tfile [open cwnd.tr w]
$tcp1 attach $tfile
$tcp1 trace cwnd_

set tcp2 [new Agent/TCP/Reno]
$ns attach-agent $n2 $tcp2
set sink2 [new Agent/TCPSink]
$ns attach-agent $n8 $sink2
$ns connect $tcp2 $sink2
$tcp2 set class_ 2
$tcp2 set packetSize_ 55
set ftp2 [new Application/FTP]
$ftp2 attach-agent $tcp2

set tfile2 [open cwnd2.tr w]
$tcp2 attach $tfile2
$tcp2 trace cwnd_

$ns at 0.5 "$ftp1 start"
$ns at 1.0 "$ftp2 start"
$ns at 5.0 "$ftp2 stop"
$ns at 5.0 "$ftp1 stop"
$ns at 5.5 "finish"

$ns run
`;
const info4 = `
  
import java.io.*;

class crc_gen {
    public static void main(String args[]) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int[] data;
        int[] div;
        int[] divisor;
        int[] rem;
        int[] crc;
        int data_bits, divisor_bits, tot_length;
        System.out.println("Enter number of data bits : ");
        data_bits = Integer.parseInt(br.readLine());
        data = new int[data_bits];
        System.out.println("Enter data bits : ");
        for (int i = 0; i < data_bits; i++)
            data[i] = Integer.parseInt(br.readLine());
        divisor_bits = 17;
        divisor = new int[]{1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1};
        tot_length = data_bits + divisor_bits - 1;
        div = new int[tot_length];
        rem = new int[tot_length];
        crc = new int[tot_length];
        for (int i = 0; i < data.length; i++)
            div[i] = data[i];
        System.out.print("Dividend (after appending 0's) are : ");
        for (int i = 0; i < div.length; i++)
            System.out.print(div[i]);
        System.out.println();
        for (int j = 0; j < div.length; j++)
            rem[j] = div[j];
        rem = divide(divisor, rem);
        for (int i = 0; i < div.length; i++)
            crc[i] = (div[i] ^ rem[i]);
        System.out.println();
        System.out.print("CRC code : ");
        for (int i = 0; i < crc.length; i++)
            System.out.print(crc[i]);
        System.out.println();
        System.out.println("Enter CRC code of " + tot_length + " bits : ");
        for (int i = 0; i < crc.length; i++)
            crc[i] = Integer.parseInt(br.readLine());
        for (int j = 0; j < crc.length; j++)
            rem[j] = crc[j];
        rem = divide(divisor, rem);
        for (int i = 0; i < rem.length; i++) {
            if (rem[i] != 0) {
                System.out.println("Error");
                break;
            }
            if (i == rem.length - 1)
                System.out.println("No Error");
        }
        System.out.println("THANK YOU. .... )");
    }

    static int[] divide(int divisor[], int rem[]) {
        int cur = 0;
        while (true) {
            for (int i = 0; i < divisor.length; i++)
                rem[cur + i] = (rem[cur + i] ^ divisor[i]);
            while (rem[cur] == 0 && cur != rem.length - 1)
                cur++;
            if ((rem.length - cur) < divisor.length)
                break;
            }
        return rem;
    }
  }
`;
const info5 = `

#include <stdio.h>

int main() {
    int w, f, frames[50];

    printf("Enter window size: ");
    scanf("%d", &w);

    printf("Enter number of frames to transmit: ");
    scanf("%d", &f);

    printf("Enter %d frames: ", f);
    for (int i = 0; i < f; i++)
        scanf("%d", &frames[i]);

    printf("\nWith sliding window protocol the frames will be sent in the following manner ");
    printf("(assuming no corruption of frames)\n\n");
    printf("After sending %d frames at each stage sender waits for acknowledgement\n\n", w);

    for (int i = 0; i < f; i++) {
        printf("%d ", frames[i]);
        if ((i + 1) % w == 0) {
            printf("\nAcknowledgement of above frames sent is received by sender\n\n");
        }
    }

    if (f % w != 0)
        printf("\nAcknowledgement of above frames sent is received by sender\n");

    return 0;
`;
const info6 = `
import numpy as np

def tsp_nearest_neighbor(distances):
    num_cities = distances.shape[0]
    visited = [False] * num_cities
    tour = []
    current_city = 0
    tour.append(current_city)
    visited[current_city] = True

    for _ in range(num_cities - 1):
        nearest_city = None
        nearest_distance = float('inf')
        for next_city in range(num_cities):
            if not visited[next_city] and distances[current_city, next_city] < nearest_distance:
                nearest_city = next_city
                nearest_distance = distances[current_city, next_city]
        current_city = nearest_city
        tour.append(current_city)
        visited[current_city] = True

    tour.append(tour[0])  # Return to starting city
    return tour

if __name__ == "__main__":
    distances = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    tour = tsp_nearest_neighbor(distances)
    print("Tour:", tour)
`;
const info7 = `
class KnowledgeBase:
    def __init__(self):
        self.known_facts = set()
        self.inference_rules = []

    def add_fact(self, fact):
        self.known_facts.add(fact)

    def add_rule(self, condition, result):
        self.inference_rules.append((condition, result))

    def forward_chaining(self, target):
        derived_facts = set()
        to_process = list(self.known_facts)

        while to_process:
            current = to_process.pop(0)
            if current == target:
                return True

            for condition, result in self.inference_rules:
                if condition in derived_facts:
                    if result not in derived_facts and result not in to_process:
                        to_process.append(result)

            derived_facts.add(current)

        return False

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_fact("A")
    kb.add_fact("B")
    kb.add_rule("A", "C")
    kb.add_rule("B", "C")
    kb.add_rule("C", "D")
    
    target_goal = "D"
    if kb.forward_chaining(target_goal):
        print(f"The goal '{target_goal}' is reachable.")
    else:
        print(f"The goal '{target_goal}' is not reachable.")
`;
const info8 = `
class Statement:
    def __init__(self, predicate_name, parameters):
        self.predicate_name = predicate_name
        self.parameters = parameters

    def __eq__(self, other):
        return isinstance(other, Statement) and self.predicate_name == other.predicate_name and self.parameters == other.parameters

    def __hash__(self):
        return hash((self.predicate_name, tuple(self.parameters)))

    def __str__(self):
        return f"{self.predicate_name}({', '.join(self.parameters)})"

    def __lt__(self, other):
        if not isinstance(other, Statement):
            return NotImplemented
        if self.predicate_name < other.predicate_name:
            return True
        elif self.predicate_name == other.predicate_name:
            return self.parameters < other.parameters
        else:
            return False

class Rule:
    def __init__(self, statements):
        self.statements = set(statements)

    def __eq__(self, other):
        return isinstance(other, Rule) and self.statements == other.statements

    def __hash__(self):
        return hash(tuple(sorted(self.statements)))

    def __str__(self):
        return " | ".join(str(stmt) for stmt in self.statements)

def apply_resolution(rule1, rule2):
    new_rules = set()
    for stmt1 in rule1.statements:
        for stmt2 in rule2.statements:
            if stmt1.predicate_name == stmt2.predicate_name and stmt1.parameters != stmt2.parameters:
                merged_statements = (rule1.statements | rule2.statements) - {stmt1, stmt2}
                new_rules.add(Rule(merged_statements))
    return new_rules

def resolution_process(knowledge_base, goal):
    pending_rules = list(knowledge_base)
    while pending_rules:
        current = pending_rules.pop(0)
        for existing in list(knowledge_base):
            if current != existing:
                new_generated = apply_resolution(current, existing)
                for new_rule in new_generated:
                    if new_rule not in knowledge_base:
                        pending_rules.append(new_rule)
                        knowledge_base.add(new_rule)
                    if not new_rule.statements:
                        return True
                    if goal in new_rule.statements:
                        return True
    return False

if __name__ == "__main__":
    kb = {
        Rule({Statement("P", ["a", "b"]), Statement("Q", ["a"])}),
        Rule({Statement("P", ["x", "y"])}),
        Rule({Statement("Q", ["y"]), Statement("R", ["y"])}),
        Rule({Statement("R", ["z"])}),
    }

    target = Statement("R", ["a"])
    found = resolution_process(kb, target)

    if found:
        print("Query is satisfiable.")
    else:
        print("Query is unsatisfiable.")
`;
const info9 = `
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            if self.check_winner(position):
                print(f"Player {self.current_player} wins!")
                return True
            elif ' ' not in self.board:
                print("It's a tie!")
                return True
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
                return False
        else:
            print("That position is already taken!")
            return False

    def check_winner(self, position):
        row_index = position // 3
        col_index = position % 3
        # Check row
        if all(self.board[row_index*3 + i] == self.current_player for i in range(3)):
            return True
        # Check column
        if all(self.board[col_index + i*3] == self.current_player for i in range(3)):
            return True
        # Check diagonal
        if row_index == col_index and all(self.board[i*3 + i] == self.current_player for i in range(3)):
            return True
        # Check anti-diagonal
        if row_index + col_index == 2 and all(self.board[i*3 + (2-i)] == self.current_player for i in range(3)):
            return True
        return False

def main():
    game = TicTacToe()
    while True:
        game.print_board()
        position = int(input(f"Player {game.current_player}, enter your position (0-8): "))
        if game.make_move(position):
            game.print_board()
            break

if __name__ == "__main__":
    main()
`;
app.get('/info1', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info1);   // return the Python code
});
app.get('/info2', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info2);   // return the Python code
});
app.get('/info3', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info3);   // return the Python code
});
app.get('/info4', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info4);   // return the Python code
});
app.get('/info5', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info5);   // return the Python code
});
app.get('/info6', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info6);   // return the Python code
});
app.get('/info7', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info7);   // return the Python code
});
app.get('/info8', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info8);   // return the Python code
});
app.get('/info9', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info9);   // return the Python code
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
module.exports = app;
