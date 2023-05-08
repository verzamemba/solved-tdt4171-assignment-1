Download Link: https://assignmentchef.com/product/solved-tdt4171-assignment-1
<br>
<h1>Problem 1</h1>

The probability that a person has 0, 1, 2, 3, 4, or 5 or more siblings is 0.15, 0.49, 0.27, 0.06, 0.02,

0.01, respectively.

<ol>

 <li>What is the probability that a child has at most 2 siblings?</li>

 <li>What is the probability that a child has more than 2 siblings given that he has at least 1 sibling?</li>

 <li>Three friends who are not siblings are gathered. What is the probability that they combined have three siblings?</li>

 <li>Emma and Jacob are not siblings, but combined they have a total of 3 siblings. What is the probability that Emma has no siblings?</li>

</ol>

<h1>Problem 2</h1>

Given the Bayesian network structure below, decided whether the statements are true or false. Justify each answer with an explaination.

<ol>

 <li>If every variable in the network has a Boolean state, then the Bayesian network can be represented with 18 numbers.</li>

 <li><em>G </em>⊥⊥ <em>A</em></li>

 <li><em>E </em>⊥⊥ <em>H </em>| {<em>D,G</em>}</li>

 <li><em>E </em>⊥⊥ <em>H </em>| {<em>C,D,F</em>}</li>

</ol>

<h1>Problem 3</h1>

The Bayesian network below contains only binary states. The conditional probability for each state is listed. From the Bayesian network, calculate the following probabilities:

<em>P</em>(<em>d </em>| <em>b</em>) = 0<em>.</em>6

<em>P</em>(¬<em>d </em>| <em>b</em>) = 0<em>.</em>4

<em>P</em>(<em>d </em>| ¬<em>b</em>) = 0<em>.</em>8

<em>P</em>(¬<em>d </em>| ¬<em>b</em>) = 0<em>.</em>2

<em>P</em>(<em>c </em>| <em>b</em>) = 0<em>.</em>1

<em>P</em>(<em>b </em>| <em>a</em>) = 0<em>.</em>5              <em>P</em>(¬<em>c </em>| <em>b</em>) = 0<em>.</em>9 <em>P</em>(¬<em>b </em>| <em>a</em>) = 0<em>.</em>5     <em>P</em>(<em>c </em>| ¬<em>b</em>) = 0<em>.</em>3

<em>P</em>(<em>b </em>| ¬<em>a</em>) = 0<em>.</em>2                      <em>P</em>(¬<em>c </em>| ¬<em>b</em>) = 0<em>.</em>7

<em>P</em>(¬<em>b </em>| ¬<em>a</em>) = 0<em>.</em>8

<ol>

 <li><em>P</em>(<em>b</em>)</li>

 <li><em>P</em>(<em>d</em>)</li>

 <li><em>P</em>(<em>c </em>| ¬<em>d</em>)</li>

 <li><em>P</em>(<em>a </em>| ¬<em>c,d</em>)</li>

</ol>

<h1>Problem 4</h1>

For this exercise you are going to implement inference by enumeration for Bayesian neural networks. The algorithm is detailed in Figure 14.9 of <em>Artificial Intelligence: A Modern Approach </em>[1]. It is completely up to you what programming language to use, but make sure your code is readable. This includes sensible variable and function names, and comments where appropriate.

We have provided a Python file implementing parts of this exercise. You are free to use this as is, and implement only the remaining parts. It is not a requirement to use the provided code; you can refactor it or disregard it if you wish. In the event you are not using the provided code, please provide a clear description of the environment you are using to run the code, including version of programming language, packages, and other details needed to run your code.

<strong>Make sure to read the whole exercise before you start coding.</strong>

<ol>

 <li>First you need to implement a Bayesian network that supports discrete conditional probability distributions for each state. This includes

  <ul>

   <li>A discrete conditional probability table (CPT). This class should have the following functionality: Probability(event, evidence), a function that takes in a variable state and the state of the variable’s parents, and returns the conditional probability of that event given the evidence. This function is not meant to calculate anything. It is only meant to look up the probability in the CPT. (This part is implemented in the provided code. See class Variable.) A directed acyclic graph (DAG) to represent the network topology. A Bayesian network is a DAG of variables. It is up to you how you want to implement the DAG, but you will need to be able to add variables and edges to the network. You will also need to be able to provide a topological ordering of the nodes (see below). (This is implemented in the provided code. See class BayesianNetwork.)</li>

   <li>A topological ordering of nodes. Topological ordering is required for First(<em>vars</em>) in EnumerateAll to work. A topological ordering ensures that each selected node has all its parent states fixed. You can use any method you choose to select nodes in correct order, but a tip is to use <em>Kahn’s algorithm </em>to put the nodes in a sorted list before iterating through them. You can find the algorithm described here: <a href="https://en.wikipedia.org/wiki/Topological_sorting">https://en.wikipedia.org/wiki/Topological_sorting</a></li>

  </ul></li>

</ol>

<strong>If you choose to use the provided code, the topological ordering of the variables is the only part of this exercise (4a) you will need to implement.</strong>

<ol start="14">

 <li>Implement the inference by enumeration algorithm found in Figure 14.9 of <em>Artificial Intelligence: A Modern Approach </em>[1]. We have also provided an outline of the algorithm below.</li>

</ol>

NB: When using a mutable type (list, dict, etc.) in python a pointer, and not a copy of the object is passed as arguments in a function. To ensure correct behaviour in recursive algorithms you will often have to copy the object with the <em>.copy() </em>method.

<table width="571">

 <tbody>

  <tr>

   <td width="384"><strong>Algorithm 1 </strong>Inference by Enumeration</td>

   <td width="187"> </td>

  </tr>

  <tr>

   <td width="384"><strong>function </strong>Enumeration-Ask(<em>X,</em><em>e,bn</em>)<em>Q</em>(<em>X</em>) ← a distribution over <em>X</em>, initially empty<strong>for each </strong>value <em>x<sub>i </sub></em>of <em>X </em><strong>do</strong><em>Q</em>(<em>x<sub>i</sub></em>) ← Enumerate-All(<em>bn.</em>Vars<em>,</em><em>e<sub>x</sub></em><em><sub>i</sub></em>) <strong>return </strong>Normalize(<em>Q</em>(<em>X</em>))</td>

   <td width="187"><em>. </em><em>e<sub>x</sub></em><em><sub>i </sub></em>is <em>e </em>extended with <em>X </em>= <em>x<sub>i</sub></em></td>

  </tr>

 </tbody>

</table>

<strong>function </strong>Enumerate-All(<em>vars,</em><em>e</em>) <strong>if </strong>Empty?(<em>vars</em>) <strong>then return </strong>1<em>.</em>0

<em>Y </em>← First(<em>vars</em>) <strong>if </strong><em>Y </em>has value <em>y </em>in <em>e </em><strong>then return </strong><em>P</em>(<em>y </em>| <em>parents</em>(<em>Y </em>))× Enumerate-All(Rest(<em>vars</em>), <em>e</em>) <strong>else return </strong><sup>P</sup><em><sub>y </sub>P</em>(<em>y </em>| <em>parents</em>(<em>Y </em>))× Enumerate-All(Rest(<em>vars</em>, <em>e<sub>y</sub></em>))

<em>. </em><em>e<sub>y </sub></em>is <em>e </em>extended with <em>Y </em>= <em>y</em>

<ol>

 <li>The Monty Hall problem is a probability puzzle based on the American TV-show <em>Let’s Make a Deal</em>, and named after its original host Monty Hall. The puzzle goes as follows:</li>

</ol>

Suppose you’re on a game show, and you’re given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what’s behind the doors, opens another door, say No. 3, which has a goat. He then says to you, ”Do you want to pick door No. 2?” Is it to your advantage to switch your choice?

We assume that the game show hosts always acts according to the following rules:

<ol>

 <li>The host must always open a door that was not picked by the contestant.</li>

 <li>The host must always open a door to reveal a goat and never the car.</li>

 <li>The host must always offer the chance to switch between the originally chosen door and the remaining closed door.</li>

</ol>

Model the Monty Hall problem as a Bayesian Network using the following states <em>ChosenByGuest</em>, <em>OpenedByHost</em>, and <em>Prize</em>. Use your implementation of inference by enumeration, and the evidence described in the problem statement to answer the question; is it to your advantage to switch your choice? Answering this question entails calculating

<em>P</em>(<em>Prize </em>| <em>ChosenByGuest </em>= 1<em>,OpenedByHost </em>= 3)

(As an example, we have implemented problem 3c) in the provided code.)