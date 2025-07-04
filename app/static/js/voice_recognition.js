document.getElementById("start-record").addEventListener("click", () => {
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];

    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: "audio/wav" });
      const formData = new FormData();
      formData.append("audio", blob, "voice.wav");

      fetch("/voice", {
        method: "POST",
        body: formData,
      })
      .then(res => res.json())
      .then(data => {
        alert(data.message);
      })
      .catch(() => {
        alert("❌ Failed to verify voice.");
      });
    };

    mediaRecorder.start();
    setTimeout(() => mediaRecorder.stop(), 3000); // 3 sec recording
  });
});
