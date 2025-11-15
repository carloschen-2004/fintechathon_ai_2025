// 客户分级分析功能
document.getElementById("analyze-btn").addEventListener("click", async () => {
  const age = Number(document.getElementById("age").value);
  const risk_pref = Number(document.getElementById("risk_pref").value);
  const capital_tier = Number(document.getElementById("capital_tier").value);
  const investment_experience = Number(document.getElementById("investment_experience").value);
  
  try {
    const res = await fetch("/client/segment", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ age, risk_pref, capital_tier, investment_experience })
    });
    
    if (!res.ok) {
      throw new Error('网络请求失败');
    }
    
    const data = await res.json();
    const resultDiv = document.getElementById("segment-result");
    
    // 设置不同客户等级的样式类
    const levelClasses = ['level-1', 'level-2', 'level-3', 'level-4', 'level-5'];
    resultDiv.className = 'card ' + levelClasses[data.level - 1];
    
    resultDiv.innerHTML = `
      <h3>${data.level_name}</h3>
      <p><strong>客户等级：</strong>${data.level}/5</p>
      <p><strong>投资风格：</strong>${data.style}</p>
      <p><strong>适合投资配置：</strong>${data.recommendation}</p>
      <p><strong>风险承受能力：</strong>${data.risk_tolerance}</p>
      <p><strong>特征匹配度：</strong>${data.match_score.toFixed(2)}%</p>
    `;
    
  } catch (error) {
    console.error('分析客户失败:', error);
    document.getElementById("segment-result").innerHTML = `<p class="error">分析客户失败，请重试</p>`;
  }
});

// 股票推荐功能
document.getElementById("recommend-btn").addEventListener("click", async () => {
  const age = Number(document.getElementById("age").value);
  const risk_pref = Number(document.getElementById("risk_pref").value);
  const capital_tier = Number(document.getElementById("capital_tier").value);
  const investment_experience = Number(document.getElementById("investment_experience").value);
  const top_k = Number(document.getElementById("top_k").value || 5);
  
  try {
    const res = await fetch("/stock/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ 
        age, 
        risk_pref, 
        capital_tier, 
        investment_experience,
        top_k 
      })
    });
    
    if (!res.ok) {
      throw new Error('网络请求失败');
    }
    
    const data = await res.json();
    const recommendationsDiv = document.getElementById("recommendations");
    recommendationsDiv.innerHTML = "";
    
    if (data.recommendations && data.recommendations.length > 0) {
      data.recommendations.forEach((r, index) => {
        const div = document.createElement("div");
        div.className = "card stock-card";
        div.setAttribute('data-stock-id', r.stock_id);
        div.innerHTML = `
          <div class="stock-header">
            <strong>${index + 1}. ${r.stock_name}</strong> <small>(${r.ticker})</small>
            <span class="score">匹配分: ${r.match_score.toFixed(2)}%</span>
          </div>
          <div class="stock-details">
            <p><strong>行业：</strong>${r.industry}</p>
            <p><strong>预期收益：</strong>${r.expected_return.toFixed(2)}%</p>
            <p><strong>风险评级：</strong>${getRiskLevelText(r.risk_level)}</p>
            <p><strong>价值因子：</strong>${r.value_factor.toFixed(3)}</p>
            <p><strong>成长因子：</strong>${r.growth_factor.toFixed(3)}</p>
            <p><strong>动量因子：</strong>${r.momentum_factor.toFixed(3)}</p>
          </div>
        `;
        recommendationsDiv.appendChild(div);
        
        // 添加点击事件，显示推荐解释
        div.addEventListener('click', () => showStockExplanation(r.stock_id));
      });
    } else {
      recommendationsDiv.innerHTML = `<p class="no-data">暂无推荐股票</p>`;
    }
    
    // 更新推荐解释区域
    document.getElementById("explanation").innerHTML = `<p>选择一支股票查看详细推荐解释</p>`;
    
  } catch (error) {
    console.error('获取推荐失败:', error);
    document.getElementById("recommendations").innerHTML = `<p class="error">获取推荐失败，请重试</p>`;
  }
});

// 显示股票推荐解释
async function showStockExplanation(stockId) {
  try {
    const explanationDiv = document.getElementById("explanation");
    explanationDiv.innerHTML = `<p class="loading">加载中...</p>`;
    
    const res = await fetch("/stock/explanation", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ stock_id: stockId })
    });
    
    if (!res.ok) {
      throw new Error('获取解释失败');
    }
    
    const data = await res.json();
    explanationDiv.className = 'card';
    explanationDiv.innerHTML = `
      <h3>推荐解释</h3>
      <p><strong>股票：</strong>${data.stock_name}</p>
      <p><strong>推荐原因：</strong>${data.reason}</p>
      <p><strong>风险匹配：</strong>${data.risk_matching}</p>
      <p><strong>预期收益分析：</strong>${data.return_analysis}</p>
      <p><strong>适合客户特点：</strong>${data.client_fit}</p>
      <p><strong>注意事项：</strong>${data.cautions}</p>
    `;
    
  } catch (error) {
    console.error('获取推荐解释失败:', error);
    document.getElementById("explanation").innerHTML = `<p class="error">获取解释失败，请重试</p>`;
  }
}

// 获取风险等级文本
function getRiskLevelText(level) {
  const levels = ['极低风险', '低风险', '中等风险', '高风险', '极高风险'];
  return levels[level - 1] || '未知风险';
}

// 原始推荐系统兼容功能
document.getElementById("legacy-btn").addEventListener("click", async () => {
  const user = Number(document.getElementById("legacy-user").value || 0);
  
  try {
    const res = await fetch("/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ user_id: user, topk: 5 })
    });
    
    if (!res.ok) {
      throw new Error('网络请求失败');
    }
    
    const data = await res.json();
    const legacyResultsDiv = document.getElementById("legacy-results");
    legacyResultsDiv.innerHTML = "";
    
    data.recommendations.forEach(r => {
      const div = document.createElement("div");
      div.className = "card legacy-card";
      div.innerHTML = `<strong>资产ID ${r.item_id}</strong> <small>(类别 ${r.category})</small>
      <p>预测匹配分: ${r.score}， 预期回报: ${r.expected_return.toFixed(3)}, 波动率: ${r.volatility.toFixed(3)}</p>
      <p><em>解释：</em>${r.explanation}</p>`;
      legacyResultsDiv.appendChild(div);
    });
    
  } catch (error) {
    console.error('获取原始推荐失败:', error);
    document.getElementById("legacy-results").innerHTML = `<p class="error">获取原始推荐失败，请重试</p>`;
  }
});
  