// app.js

let tfliteModel = null;
let audioContext = null;
let meydaAnalyzer = null;
let micStream = null;

// MFCC 전처리 파라미터
const SAMPLE_RATE = 16000;    // 모델 학습 시 샘플레이트
const FFT_SIZE = 1024;
const HOP_SIZE = 512;
const NUM_MEL_BINS = 40;
const NUM_MFCC = 40;
// 1초 분량(16000샘플)에서 얻을 수 있는 슬라이스 수
const MFCC_SLICES = Math.floor((SAMPLE_RATE - FFT_SIZE) / HOP_SIZE) + 1; // ≈30

// HTML 요소
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");

// MFCC 버퍼: 슬라이스가 쌓이면 예측 수행
let mfccBuffer = [];

/**
 * 1) TFLite WASM 경로 설정 & 모델 로드
 */
async function loadTFLiteModel() {
  statusEl.innerText = "WASM 모듈 초기화 중…";

  // tfjs-tflite.js가 로드된 뒤에만 tflite 변수가 정의됩니다.
  // 반드시 여기서 setWasmPath()를 호출하여, .wasm 파일 위치를 명시해야 Module._malloc 오류가 안 납니다.
  // 버전에 맞춰 URL을 복사해주세요.
  tflite.setWasmPath(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1/dist/tfjs-tflite.wasm"
  );

  try {
    statusEl.innerText = "TFLite 모델 로딩 중…";
    // 모델을 로드할 때 await를 사용해서, 완전히 초기화된 뒤에만 다음 코드가 실행되게 함
    tfliteModel = await tflite.loadTFLiteModel("baby_cry_model.tflite");
    console.log("tfliteModel:", tfliteModel);
    statusEl.innerText = "모델 로드 완료! 마이크 시작 가능.";
    startBtn.disabled = false;
  } catch (e) {
    console.error("모델 로드 실패:", e);
    statusEl.innerText = "모델 로드 실패!";
  }
}

// 페이지 DOMContentLoaded 이벤트 후 모델 로드 시작
window.addEventListener("DOMContentLoaded", () => {
  loadTFLiteModel();
});

/**
 * 2) 마이크 시작
 */
async function startMicrophone() {
  // 모델이 준비되지 않았다면 경고 후 리턴
  if (tfliteModel === null) {
    alert("모델 로드가 아직 완료되지 않았습니다.");
    return;
  }
  if (audioContext) return; // 이미 실행 중이면 아무 작업 안 함

  // AudioContext 생성
  audioContext = new (window.AudioContext || window.webkitAudioContext)();

  // 마이크 권한 요청
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.innerText = "마이크 접근 실패: " + err.message;
    return;
  }

  const sourceNode = audioContext.createMediaStreamSource(micStream);

  // Meyda Analyzer 설정
  meydaAnalyzer = Meyda.createMeydaAnalyzer({
    audioContext: audioContext,
    source: sourceNode,
    bufferSize: FFT_SIZE,
    hopSize: HOP_SIZE,
    sampleRate: SAMPLE_RATE, // Meyda 내부에서 리샘플링
    featureExtractors: ["mfcc"],
    numberOfMFCCCoefficients: NUM_MFCC,
    melBands: NUM_MEL_BINS,
    callback: onMeydaFeatures,
  });

  meydaAnalyzer.start();
  statusEl.innerText = "마이크 녹음 중…";
  startBtn.disabled = true;
  stopBtn.disabled = false;
}

/**
 * 3) Meyda 콜백: MFCC 추출 → 버퍼링 → TFLite 예측
 */
async function onMeydaFeatures(features) {
  if (!features || !features.mfcc) return;
  // 모델이 로드되지 않았다면 예측 로직을 건너뜀
  if (tfliteModel === null) return;

  // 40차원 MFCC를 mfccBuffer 배열에 이어붙임
  mfccBuffer.push(...features.mfcc);

  // 충분한 슬라이스가 모였으면 예측 수행
  if (mfccBuffer.length >= MFCC_SLICES * NUM_MFCC) {
    // 1D Float32Array로 변환
    const inputArray = new Float32Array(
      mfccBuffer.slice(0, MFCC_SLICES * NUM_MFCC)
    );
    try {
      // TFLite 예측: outputTensor는 Float32Array 형태
      const outputTensor = tfliteModel.predict(inputArray);
      const prob = outputTensor[0]; // 예: [0] 번째가 울음 확률
      resultEl.innerText = `울음 확률: ${prob.toFixed(3)}`;
    } catch (err) {
      console.error("예측 오류:", err);
    }
    // 롤링 윈도우: hop 만큼만 앞부분 잘라서 남김
    const hopSliceCount = Math.floor((HOP_SIZE / FFT_SIZE) * MFCC_SLICES);
    mfccBuffer = mfccBuffer.slice(hopSliceCount * NUM_MFCC);
  }
}

/**
 * 4) 마이크 중지
 */
function stopMicrophone() {
  if (meydaAnalyzer) {
    meydaAnalyzer.stop();
    meydaAnalyzer = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
  statusEl.innerText = "마이크 중지됨";
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// 버튼 이벤트 연결
startBtn.addEventListener("click", startMicrophone);
stopBtn.addEventListener("click", stopMicrophone);
