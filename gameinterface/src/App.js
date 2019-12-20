import React from 'react';
import './App.css';

const BOARD_RADIUS = 3;
const BOARD_SIZE = 2 * BOARD_RADIUS + 1;
const BOARD_SCALE_PX = 50;

const ALL_VALID_QR = [
    [0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 3],
    [1, 4], [1, 5], [1, 6], [2, 1], [2, 2], [2, 3],
    [2, 4], [2, 5], [2, 6], [3, 0], [3, 1], [3, 2],
    [3, 3], [3, 4], [3, 5], [3, 6], [4, 0], [4, 1],
    [4, 2], [4, 3], [4, 4], [4, 5], [5, 0], [5, 1],
    [5, 2], [5, 3], [5, 4], [6, 0], [6, 1], [6, 2],
    [6, 3],
];

const ALL_SCORING_QR = [
    [0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6],
    [2, 1], [2, 6], [3, 0], [3, 3], [3, 6], [4, 0],
    [4, 5], [5, 0], [5, 4], [6, 0], [6, 1], [6, 2],
    [6, 3],
];

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
                fill={{0: 'darkgrey', '1': '#66f', '2': '#f66', '3': '#99f', '4': '#f99'}[this.cells[qr]]}
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
            connected: false,
            disconnectionTimeout: undefined,
        };
        this.ws = new WebSocket("ws://localhost:8765/api");
        this.ws.onopen = function() {
            //alert('Open');
        };
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
                    this.state.board.playerToMove = data.playerToMove;
                    this.state.board.playerMoveIndex = data.playerMoveIndex;
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
        /*
        if (this.state.board.cells[qr] != 0)
            return;
        this.state.board.cells[qr] = this.state.board.playerToMove;
        if (this.state.board.playerMoveIndex === 'a') {
            this.state.board.playerMoveIndex = 'b';
        } else {
            this.state.board.playerToMove = 3 - this.state.board.playerToMove;
            this.state.board.playerMoveIndex = 'a';
        }
        this.forceUpdate();
        */
        this.ws.send(JSON.stringify({
            kind: 'click', qr
        }));
    }

    render() {
        const toMove = this.state.board.playerToMove == 1 ? 'blue' : 'red';
        return <div style={{margin: '5px'}} >
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
                    Turn: <span style={{color: toMove}}>{toMove}</span> - {
                        {a: 'first', b: 'second', win: 'winner'}[this.state.board.playerMoveIndex]
                    }
                </div>
                <div style={{opacity: this.state.connected ? 1 : 0.3}}>
                    {this.state.board.renderSVG(this.onClick)}
                </div>
            </div>
        </div>;
    }
}

export default App;
