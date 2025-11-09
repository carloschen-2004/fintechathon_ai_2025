document.getElementById("btn").addEventListener("click", async () => {
    const user = Number(document.getElementById("user").value || 0);
    const k = Number(document.getElementById("k").value || 5);
    const res = await fetch("/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ user_id: user, topk: k })
    });
    const data = await res.json();
    const list = document.getElementById("rec-list");
    list.innerHTML = "";
    data.recommendations.forEach(r => {
      const div = document.createElement("div");
      div.className = "card";
      div.innerHTML = `<strong>资产ID ${r.item_id}</strong> <small>(类别 ${r.category})</small>
      <p>预测匹配分: ${r.score}， 预期回报: ${r.expected_return.toFixed(3)}, 波动率: ${r.volatility.toFixed(3)}</p>
      <p><em>解释：</em>${r.explanation}</p>`;
      list.appendChild(div);
    });
  });
  