console.time("connecting");
console.time("loading");
// image stuff
let last_prompt = null;
let last_seed = null;
let last_sent = null;

// async function testStability {
//     var
//     fetch("")
//     //https://api.stability.ai/v1alpha/generation/stable-diffusion-512-v2-0/text-to-image"  -H 'Content-Type: application/json'  -H 'Accept: image/png' -H "Authorization: sk-2ZP1rHM6OsQzzedLAkeHzPiZeAd5PYAWek35JqKpPsWsgitW"  --data-raw '{"cfg_scale": 7,"clip_guidance_preset": "NONE","height": 512,"width": 512,"samples": 1,"seed": 0,"steps": 50,"text_prompts": [{"text": "A lighthouse on a cliff","weight": 1}]}'
// }

async function getPrompt() {
  var prompt = document.getElementById("prompt");
  var seed = document.getElementById("seed");
  while (true) {
    console.log("checking if prompt");
    if (prompt && prompt.value) {
      if (prompt.value !== last_prompt || seed.value !== last_seed) {
        last_prompt = prompt.value;
        last_seed = seed;
        console.log("got prompt");
        last_sent = Date.now();
        console.time("generation");
        return JSON.stringify({ prompt: prompt.value, seed: seed.value, id: last_sent });
      }
    }
    await new Promise((r) => setTimeout(r, 100));
  }
}


function handleImage(data) {
  waiting = false;
  console.log("handling image");
  var parsed = JSON.parse(data);
  var latency = Math.round(Date.now() - parsed.id);
  var latencyField = document.getElementById("latency");
  latencyField.textContent = `latency: ${latency}ms`;
  document.getElementById("gen_time").textContent = `generation: ${parsed.gen_time}ms`;
  var top = document.getElementById("imoge");
  var bottom = document.getElementById("imoge2");
  if (top.style.opacity == 1) {
    bottom.src = parsed.image;
    bottom.style.opacity = 1;
    top.style.opacity = 0;
  } else {
    top.src = parsed.image;
    top.style.opacity = 1;
    bottom.style.opacity = 0;
  }
  sendPrompt();
}

var sending = false;
var waiting = false;
function sendPrompt() {
  if (waiting) {
      console.log("already waiting, not sending again")
      return;
  }
  sending = true;
  getPrompt().then((prompt) => {
    let interval = setInterval(function () {
        if (dc !== null && dc_open) {
          console.log("got prompt, actually sending over rtc");
          dataChannelLog.textContent += "> " + prompt + "\n";
          dc.send(prompt);
          clearInterval(interval);
          sending = false;
          waiting = true;
        } else if (ws && ws.readyState === 1) {
          console.log("sending over ws");
          ws.send(prompt);
          clearInterval(interval);
          sending = false;
          waiting = true;
        } else {
          console.log("no connections open, retrying");
        }
    }, 1000);
  });
}

// webrtc stuff

// get DOM elements
var dataChannelLog = document.getElementById("data-channel"),
  iceConnectionLog = document.getElementById("ice-connection-state"),
  iceGatheringLog = document.getElementById("ice-gathering-state"),
  signalingLog = document.getElementById("signaling-state");

// peer connection
var pc = null;

// data channel
var dc = null,
  dcInterval = null;
var dc_open = false;

function createPeerConnection() {
  var config = {
    sdpSemantics: "unified-plan",
  };

  if (document.getElementById('use-stun').checked) {
  // hm
     config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
  }

  pc = new RTCPeerConnection(config);

  // register some listeners to help debugging
  pc.addEventListener(
    "icegatheringstatechange",
    function () {
      iceGatheringLog.textContent += " -> " + pc.iceGatheringState;
    },
    false
  );
  iceGatheringLog.textContent = pc.iceGatheringState;

  pc.addEventListener(
    "iceconnectionstatechange",
    function () {
      iceConnectionLog.textContent += " -> " + pc.iceConnectionState;
    },
    false
  );
  iceConnectionLog.textContent = pc.iceConnectionState;

  pc.addEventListener(
    "signalingstatechange",
    function () {
      signalingLog.textContent += " -> " + pc.signalingState;
    },
    false
  );
  signalingLog.textContent = pc.signalingState;

  // connect audio / video
  // pc.addEventListener('track', function(evt) {
  //     if (evt.track.kind == 'video')
  //         document.getElementById('video').srcObject = evt.streams[0];
  //     else
  //         document.getElementById('audio').srcObject = evt.streams[0];
  // });
  // sdpFilterCodec

  return pc;
}

function negotiate() {
  return pc
    .createOffer()
    .then(function (offer) {
      return pc.setLocalDescription(offer);
    })
    .then(function () {
      // wait for ICE gathering to complete
      return new Promise(function (resolve) {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          function checkState() {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", checkState);
              resolve();
            }
          }
          pc.addEventListener("icegatheringstatechange", checkState);
        }
      });
    })
    .then(function () {
      var offer = pc.localDescription;
      document.getElementById("offer-sdp").textContent = offer.sdp;
      // this part needs to go through runpod
      // proxy is fine-ish for this
      return fetch("/offer", {
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    })
    .then(function (response) {
      return response.json();
    })
    .then(function (answer) {
      document.getElementById("answer-sdp").textContent = answer.sdp;
      return pc.setRemoteDescription(answer);
    })
    .catch(function (e) {
      console.log(e);
      alert(e);
    });
}

var time_start = null;

function current_stamp() {
  if (time_start === null) {
    time_start = new Date().getTime();
    return 0;
  } else {
    return new Date().getTime() - time_start;
  }
}

function start() {
  pc = createPeerConnection();

  // {"ordered": false, "maxRetransmits": 0}

  // {"ordered": false, "maxPacketLifetime": 500}
  dc = pc.createDataChannel("chat", { ordered: true });
  dc.onclose = function () {
    dc_open = false;
    clearInterval(dcInterval);
    dataChannelLog.textContent += "- close\n";
  };
  dc.onopen = function () {
    dc_open = true;
    console.log("onopen");
    dataChannelLog.textContent += "- open\n";
    dcInterval = setInterval(function () {
      var message = "ping " + current_stamp();
      dataChannelLog.textContent += "> " + message + "\n";
      dc.send(message);
    }, 1000);
    sendPrompt();
    console.log("started sending prompt");
    console.timeEnd("connecting");
  };
  dc.onmessage = function (evt) {
    dataChannelLog.textContent += "< " + evt.data + "\n";
    if (evt.data.substring(0, 4) === "pong") {
      var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
      dataChannelLog.textContent += " RTT " + elapsed_ms + " ms\n";
    }
    if (evt.data.substring(0, 1) === "{") {
      handleImage(evt.data);
    }
  };

  negotiate();

  document.getElementById("stop").style.display = "inline-block";
}

function stop() {
  document.getElementById("stop").style.display = "none";

  // close data channel
  if (dc) {
    dc.close();
  }

  // close transceivers
  if (pc.getTransceivers) {
    pc.getTransceivers().forEach(function (transceiver) {
      if (transceiver.stop) {
        transceiver.stop();
      }
    });
  }

  // close local audio / video
  pc.getSenders().forEach(function (sender) {
    sender.track.stop();
  });

  // close peer connection
  setTimeout(function () {
    pc.close();
  }, 500);
}

let ws = new WebSocket(
  (window.location.protocol === "https:" ? "wss://" : "ws://") +
    window.location.host +
    "/ws"
);
var ws_open = false;
setInterval(function () {
  if (ws.readyState === 1) {
    var message = "ping " + current_stamp();
    ws.send(message);
  }
}, 1000);
ws.addEventListener("open", (event) => {
  ws_open = true;
  if (!dc_open) {
    sendPrompt();
  }
});
ws.addEventListener("message", ({ data }) => {
  if (data.substring(0, 4) === "pong") {
    var elapsed_ms = current_stamp() - parseInt(data.substring(5), 10);
    console.log("ws RTT " + elapsed_ms + " ms\n");
  } else {
    handleImage(data);
  }
});
ws.addEventListener("close", (event) => {
  console.log("ws closed");
  ws_open = false;
});

//new Promise((r) => setTimeout(r, 10000)).then(() =>
start();
//);
console.timeEnd("loading");


//setInterval(function () {
//  if (!sending && !waiting) { sendPrompt() }
//}, 1000);
