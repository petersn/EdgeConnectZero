import React from 'react';
import './App.css';
//import Plot from 'react-plotly.js';
import { LineChart, Line, CartesianGrid, XAxis, YAxis } from 'recharts';

if (!String.prototype.trim) {
    String.prototype.trim = function () {
        return this.replace(/^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, '');
    };
}

// From: https://stackoverflow.com/questions/3733227/javascript-seconds-to-minutes-and-seconds
function fancyTimeFormat(time) {
    // Hours, minutes and seconds
    var hrs = ~~(time / 3600);
    var mins = ~~((time % 3600) / 60);
    var secs = ~~time % 60;

    // Output like "1:01" or "4:03:59" or "123:03:59"
    var ret = "";

    if (hrs > 0) {
        ret += "" + hrs + ":" + (mins < 10 ? "0" : "");
    }

    ret += "" + mins + ":" + (secs < 10 ? "0" : "");
    ret += "" + secs;
    return ret;
}

const STARTING_TIME = 5 * 60;
const INCREMENT = 10;

const BOARD_RADIUS = 11;
const BOARD_SIZE = 2 * BOARD_RADIUS + 1;
const BOARD_SCALE_PX = 50;

const ALL_VALID_QR = [];
const ALL_SCORING_QR = [];

for (let q = 0; q < BOARD_SIZE; q++) {
	for (let r = 0; r < BOARD_SIZE; r++) {
		if (q + r < BOARD_RADIUS)
			continue;
		if ((BOARD_SIZE - q - 1) + (BOARD_SIZE - r - 1) < BOARD_RADIUS)
			continue;
		ALL_VALID_QR.push([q, r]);
	}
}

for (let i = 0; i < BOARD_SIZE; i++) {
	ALL_SCORING_QR.push([i, 0]);
	ALL_SCORING_QR.push([0, i]);
	ALL_SCORING_QR.push([i, BOARD_SIZE - 1]);
	ALL_SCORING_QR.push([BOARD_SIZE - 1, i]);
	ALL_SCORING_QR.push([i, BOARD_RADIUS - i]);
	ALL_SCORING_QR.push([BOARD_SIZE - i - 1, BOARD_SIZE - (BOARD_RADIUS - i) - 1]);
}
ALL_SCORING_QR.push([BOARD_RADIUS, BOARD_RADIUS]);

function isScoringQr(qr) {
    for (let testQr of ALL_SCORING_QR)
        if (qr[0] == testQr[0] && qr[1] == testQr[1])
            return true;
    return false;
}

const funky_constant = 1.5 / Math.sqrt(3);

function qrToXy(qr) {
    return {x: qr[0] + 0.5 * qr[1], y: funky_constant * qr[1]};
}

function nextColor(x) {
    return (x + 1) % 3;
}

class Board {
    constructor() {
        this.playerToMove = 1;
        this.playerMoveIndex = 'a';
        this.cells = {};
        //this.refs = {};
        for (let qr of ALL_VALID_QR) {
            this.cells[qr] = 0;
            //this.refs[qr] = React.createRef();
        }
    }

    renderQR = (qr, onClick) => {
        const {x, y} = qrToXy(qr);
        const h = 0.36;
        return <>
            <path
                d={`M ${x - BOARD_RADIUS / 2 + 0.05} ${y + funky_constant / 2} l 0 0.5  0.5 ${h}  0.5 ${-h}  0 -0.5  -0.5 ${-h} Z`}
                stroke='black'
                strokeWidth='0.05'
                fill={{0: 'darkgrey', '1': '#f66', '2': '#66f', '3': '#f99', '4': '#99f', '5': '#fcc', '6': '#ccf'}[this.cells[qr]]}
                onClick={(evt) => onClick(evt, qr)}
            />
            {isScoringQr(qr) &&
                <circle
                    cx={`${x - BOARD_RADIUS / 2 + 0.5 + 0.05}`}
                    cy={`${y + funky_constant / 2 + 0.25}`} 
                    r='0.1'
                    style={{pointerEvents: 'None'}}
                />
            }
        </>;
    }

    renderSVG(onClick) {
        return <svg style={{
            width: BOARD_SCALE_PX * BOARD_SIZE + 5,
            height: BOARD_SCALE_PX * (BOARD_SIZE + 0.5) * funky_constant + 5,
            /* border: '1px solid black', */
        }}>
            <g transform={`scale(${BOARD_SCALE_PX})`}>
                {ALL_VALID_QR.map((qr) => this.renderQR(qr, onClick))}
            </g>
        </svg>;
    }
};

class App extends React.Component {
    constructor() {
        super();
        this.state = {
            board: new Board(),
            aiPlayer: '???',
            boardString: '???',
            isThinking: '???',
            scoreInterval: {1: 0, 2: 0},
            boardStringStack: [],
            nps: '???',
            evaluationCurve: {},
            chartData: [],
            timeBanks: {1: STARTING_TIME, 2: STARTING_TIME},
            clockActive: false,
            connected: false,
            disconnectionTimeout: undefined,
        };
        this.ws = new WebSocket("ws://localhost:8765/api");
        this.ws.onopen = function() {
            //alert('Open');
        };
        setInterval(
            () => {
                if (! this.state.clockActive)
                    return;
                const p = this.state.board.playerToMove;
                if (p === 1 || p === 2) {
                    this.state.timeBanks[p] -= 0.2;
                    this.forceUpdate();
                }
            },
            200,
        );
        setInterval(
            () => {
                this.ws.send(JSON.stringify({kind: 'ping'}));
            },
            250,
        );
        this.ws.onmessage = (evt) => {
            const data = JSON.parse(evt.data);
            switch (data.kind) {
                case 'board':
                    this.state.board.cells = data.board;
                    if (data.playerToMove !== this.state.board.playerToMove && this.state.clockActive) {
                        this.state.timeBanks[this.state.board.playerToMove] += INCREMENT;
                    }
                    this.state.board.playerToMove = data.playerToMove;
                    this.state.board.playerMoveIndex = data.playerMoveIndex;
                    this.state.aiPlayer = data.aiPlayer;
                    this.state.isThinking = data.isThinking;
                    if (
                        (data.boardString !== this.state.boardString) &&
                        (this.state.boardStringStack.length === 0 || this.state.boardStringStack[this.state.boardStringStack.length - 1] !== data.boardString)
                    ) {
                        this.state.boardStringStack = [...this.state.boardStringStack, data.boardString];
                    }
                    this.state.boardString = data.boardString;
                    this.state.scoreInterval = data.scoreInterval;
                    this.state.nps = data.nps;
                    this.state.evaluationCurve = data.evaluationCurve;
                    //console.log(this.state.evaluationCurve);
                    //let chartData = [];
                    this.state.chartData = this.state.chartData.filter(
                        (obj) => obj.value === this.state.evaluationCurve[obj.key]
                    );
                    for (let key of Object.keys(this.state.evaluationCurve)) {
                        key = Number(key);
                        //chartData = chartData.filter(
                        //    (obj) => obj.value === this.state.evaluationCurve[obj.key]
                        //);
                        //if (this.state.evaluationCurve[key] !== this.state.chartData[key]) 
                        if (! this.state.chartData.find((obj) => obj.key === key)) {
                            this.state.chartData.push({
                                key,
                                value: this.state.evaluationCurve[key],
                            });
                        }
                    }
                    this.state.chartData.sort((a, b) => Number(a.key) - Number(b.key));
                    break;
                case 'ping':
                    break;
                default:
                    alert('Unhandled command: ' + evt.data);
            }
            this.resetDisconnectionTimeout();
        };
    }

    resetDisconnectionTimeout() {
        clearTimeout(this.state.disconnectionTimeout);
        this.setState({
            connected: true,
            disconnectionTimeout: setTimeout(this.disconnectedHandler, 1000),
        });
    }

    disconnectedHandler = () => {
        this.setState({connected: false});
    }

    onClick = (evt, qr) => {
        this.ws.send(JSON.stringify({
            kind: 'click', qr
        }));
    }

    onResume = (evt) => {
        const resumeBoardString = this.resumeRef.value.trim();
        if (resumeBoardString.length != 403) {
            alert('Bad resume string length: ' + resumeBoardString.length + ' (Should have been 403.)');
            return;
        }
        console.log("Resuming from:", resumeBoardString);
        this.ws.send(JSON.stringify({
            kind: 'resume',
            boardString: resumeBoardString,
        }));
    }

    onPromptAIMove = (evt) => {
        this.ws.send(JSON.stringify({kind: 'genmove'}));
    }

    onUndo = (evt) => {
        if (this.state.boardStringStack.length <= 1) {
            alert('Cannot undo any further!');
            return;
        }
        const resumeBoardString = this.state.boardStringStack.pop();
        console.log("Resuming from:", resumeBoardString);
        // Anticipate the board string change, to prevent an additional undo entry being inserted.
        this.ws.send(JSON.stringify({
            kind: 'resume',
            boardString: resumeBoardString,
        }));
        this.forceUpdate();
    }

    onToggleClock = (evt) => {
        this.state.clockActive = ! this.state.clockActive;
    }

    handleKeydown = (e) => {
        if (!e) return;
        if (e.ctrlKey || e.shiftKey || e.metaKey || e.altKey) return;
        switch (e.key.toLowerCase()) {
          case 't':
            if (this.buttonRef) {
              this.buttonRef.focus();
              this.buttonRef.click();
            }
            break;
        }
      };

    componentDidMount() {
        window.addEventListener('keydown', this.handleKeydown);
    }

    componentWillUnmount() {
        window.removeEventListener('keydown', this.handleKeydown);
    }

    render() {
        const toMove = this.state.board.playerToMove == 1 ? 'red' : 'blue';
        let winBelief = 0;
        let largestKey = -1;
        for (let key of Object.keys(this.state.evaluationCurve)) {
            key = Number(key);
            if (key > largestKey)
                largestKey = key;
        }
        if (largestKey !== -1)
            winBelief = this.state.evaluationCurve[largestKey];
        const wrapTime = (k, s) => {
            if (this.state.board.playerToMove === k)
                return '[' + s + ']';
            return <>&nbsp;{s}&nbsp;</>;
        };

        return <div style={{
                margin: '5px',
                display: 'flex',
                justifyContent: 'space-between'
            }} >
            <div className="left" style={{ width: '50%' }}>
            <div style={{
                display: 'inline-block',
                backgroundColor: '#ddd',
                border: '2px solid black',
                borderRadius: '5px',
                marginBottom: '5px',
                padding: '10px',
                textAlign: 'center',
                fontSize: '200%',
                fontFamily: 'monospace',
                fontWeight: 'bold',
            }}>
                Clock:&nbsp;
                <span style={{color: 'red'}}>{wrapTime(1, fancyTimeFormat(this.state.timeBanks[1]).padStart(4))}</span>&nbsp;
                <span style={{color: 'blue'}}>{wrapTime(2, fancyTimeFormat(this.state.timeBanks[2]).padStart(4))}</span>
                {/*
                <span style={{color: 'red'}}>{this.state.timeBanks[1].toFixed(1).padStart(5)}</span>&nbsp;
                <span style={{color: 'blue'}}>{this.state.timeBanks[2].toFixed(1).padStart(5)}</span>
                */}
            </div><br/>

            <div style={{
                display: 'inline-block',
                backgroundColor: '#ddd',
                border: '2px solid black',
                borderRadius: '5px',
                padding: '10px',
                textAlign: 'center',
                fontSize: '200%',
                fontFamily: 'monospace',
                fontWeight: 'bold',
            }}>
                <div style={{marginBottom: '5px'}}>
                    {this.state.isThinking > 0 ? 'Thinking... ' : ''}
                    Turn: <span style={{color: toMove}}>{toMove}</span> - {
                        {a: 'first', b: 'second', win: 'winner'}[this.state.board.playerMoveIndex]
                    }
                </div>
                <div style={{opacity: this.state.connected ? 1 : 0.3}}>
                    {this.state.board.renderSVG(this.onClick)}
                </div>
            </div>
            </div>
            {/*
            <Plot
                data={[
                    {
                        x: Object.keys(this.state.evaluationCurve),
                        y: Object.values(this.state.evaluationCurve),
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: {color: 'red'},
                    },
                ]}
                layout={{width: 320, height: 240, title: "AI's belief that it will win"}}
            />
            */}
            <div className="right" style={{ width: '50%' }}>
                <b style={{fontSize: '150%'}}>AI's belief that it will win:</b>
                <LineChart width={1000} height={500} data={this.state.chartData}>
                    <Line type="monotone" dataKey="value" stroke="teal" strokeWidth="5" />
                    <CartesianGrid stroke="#ccc" />
                    <XAxis dataKey="key" />
                    <YAxis domain={[0, 1]}/>
                </LineChart>
                <div>
                    <br/>
                    <div style={{fontSize: '130%'}}>
                        AI's current belief that it will win: {(100 * winBelief).toFixed(1)}%<br/>
                        Score interval: {this.state.scoreInterval[1]}, {this.state.scoreInterval[2]}<br/>
                        Nodes searched last turn: {Math.round(2 * this.state.nps)}<br/>
                    </div>
                    <br/>
                    <b>Debugging tools:</b><br/>
                    AI player: {this.state.aiPlayer}<br/>
                    Stack depth: {this.state.boardStringStack.length}<br/>
                    Is thinking: {this.state.isThinking}<br/>
                    Board string:<br/><br/>{this.state.boardString}<br/><br/>
                    Resume from: <input type="text" ref={(resumeRef) => { this.resumeRef = resumeRef; }}/><br/>
                    <button onClick={this.onResume}>Restore Game</button>
                    <button onClick={this.onPromptAIMove}>Prompt AI move</button>
                    <button onClick={this.onToggleClock} ref={(ele) => { this.buttonRef = ele; }}>Toggle Clock</button>
                    <button onClick={this.onUndo}>Undo</button>
                </div>
            </div>
        </div>;
    }
}

export default App;
