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
  // 获取表单数据
  const age = Number(document.getElementById("age").value);
  const risk_pref = Number(document.getElementById("risk_pref").value);
  const capital_tier = Number(document.getElementById("capital_tier").value);
  const investment_experience = Number(document.getElementById("investment_experience").value);
  const top_k = Number(document.getElementById("top_k").value || 5);
  
  // 表单验证
  if (isNaN(age) || isNaN(risk_pref) || isNaN(capital_tier) || isNaN(investment_experience)) {
    const recommendationsDiv = document.getElementById("recommendations");
    recommendationsDiv.innerHTML = `<p class="error">请填写有效的数值</p>`;
    return;
  }
  
  // 显示加载状态
  const recommendationsDiv = document.getElementById("recommendations");
  recommendationsDiv.innerHTML = `<p class="loading">加载中...</p>`;
  
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
      throw new Error('网络请求失败: ' + res.status);
    }
    
    const data = await res.json();
    recommendationsDiv.innerHTML = "";
    
    if (data.recommendations && data.recommendations.length > 0) {
      const stockList = document.createElement("div");
      stockList.className = "stock-list";
      
      data.recommendations.forEach((r, index) => {
        const div = document.createElement("div");
        div.className = "card stock-card hoverable";
        div.setAttribute('data-stock-id', r.stock_id);
        div.innerHTML = `
          <div class="stock-header">
            <div class="rank-badge">#${index + 1}</div>
            <strong>${r.stock_name}</strong> <small>(${r.ticker})</small>
            <span class="score">匹配分: ${r.match_score.toFixed(2)}%</span>
          </div>
          <div class="stock-details">
            <p><strong>行业：</strong>${r.industry || '未知'}</p>
            <p><strong>预期收益：</strong>${r.expected_return.toFixed(2)}%</p>
            <p><strong>风险评级：</strong>${getRiskLevelText(r.risk_level)}</p>
            <p><strong>价值因子：</strong>${r.value_factor.toFixed(3)}</p>
            <p><strong>成长因子：</strong>${r.growth_factor.toFixed(3)}</p>
            <p><strong>动量因子：</strong>${r.momentum_factor.toFixed(3)}</p>
            <div class="stock-actions">点击查看详情</div>
          </div>
        `;
        stockList.appendChild(div);
        
        // 添加点击事件，显示推荐解释
        div.addEventListener('click', () => {
          // 移除之前选中的样式
          document.querySelectorAll('.stock-card.selected').forEach(el => {
            el.classList.remove('selected');
          });
          // 添加选中样式
          div.classList.add('selected');
          showStockExplanation(r.stock_id);
        });
      });
      
      recommendationsDiv.appendChild(stockList);
    } else {
      recommendationsDiv.innerHTML = `<p class="no-data">暂无推荐股票</p>`;
    }
    
    // 更新推荐解释区域
    document.getElementById("explanation").innerHTML = `<p>选择一支股票查看详细推荐解释</p>`;
    
  } catch (error) {
    console.error('获取推荐失败:', error);
    recommendationsDiv.innerHTML = `<p class="error">获取推荐失败: ${error.message}，请重试</p>`;
  }
});

// 辅助函数：显示加载状态
function showLoading(element) {
  element.innerHTML = '<p class="loading">加载中...</p>';
}

// 辅助函数：显示错误信息
function showError(element, message) {
  element.innerHTML = `<p class="error">${message}</p>`;
}

// 全局状态管理
const state = {
    currentExplanation: null,
    currentSimilarStocks: null
};

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
    state.currentExplanation = data;
    explanationDiv.className = 'card';
    explanationDiv.innerHTML = `
      <h3>${data.stock_name} (${data.ticker})</h3>
      <div class="explanation-details">
        <div class="explanation-section">
          <h4>推荐原因</h4>
          <p class="reason-text">${data.reason}</p>
        </div>
        
        <div class="explanation-section">
          <h4>风险匹配分析</h4>
          <p>${data.risk_matching}</p>
          
          <div class="risk-level-indicator">
            <div class="risk-label">风险等级:</div>
            <div class="risk-bars">
              ${Array.from({ length: 5 }, (_, i) => 
                  `<div class="risk-bar ${i < data.risk_level ? 'active' : ''}"></div>`
              ).join('')}
            </div>
          </div>
        </div>
        
        <div class="explanation-section">
          <h4>收益分析</h4>
          <p>${data.return_analysis}</p>
          <p>${data.client_fit}</p>
        </div>
        
        <div class="explanation-section caution-section">
          <h4>注意事项</h4>
          <p>${data.cautions}</p>
        </div>
      </div>
    `;
    
    // 添加查看相似股票按钮
    const actionButtons = document.createElement('div');
    actionButtons.className = 'action-buttons';
    actionButtons.innerHTML = `
      <button id="view-similar-btn" class="primary-button">查看相似股票</button>
    `;
    explanationDiv.appendChild(actionButtons);
    
    // 添加按钮点击事件
    document.getElementById('view-similar-btn').addEventListener('click', function() {
      // 加载相似股票
      loadSimilarStocks(stockId);
      
      // 滚动到相似股票区域，如果存在
      const similarSection = document.getElementById('similar-stocks-section');
      if (similarSection) {
        similarSection.style.display = 'block';
        similarSection.scrollIntoView({ behavior: 'smooth' });
      }
    });
    
  } catch (error) {
    console.error('获取推荐解释失败:', error);
    document.getElementById("explanation").innerHTML = `<p class="error">获取解释失败，请重试</p>`;
  }
}

// 加载相似股票推荐
async function loadSimilarStocks(stockId) {
  // 检查是否已存在相似股票区域，如果不存在则创建
  let similarSection = document.getElementById('similar-stocks-section');
  let similarResult;
  
  if (!similarSection) {
    similarSection = document.createElement('div');
    similarSection.id = 'similar-stocks-section';
    similarSection.className = 'section';
    similarSection.style.display = 'block';
    
    const sectionTitle = document.createElement('h2');
    sectionTitle.textContent = '相似股票推荐';
    similarSection.appendChild(sectionTitle);
    
    similarResult = document.createElement('div');
    similarResult.id = 'similar-result';
    similarSection.appendChild(similarResult);
    
    // 插入到解释区域之后
    const explanationDiv = document.getElementById('explanation');
    explanationDiv.parentNode.insertBefore(similarSection, explanationDiv.nextSibling);
  } else {
    similarResult = document.getElementById('similar-result');
  }
  
  similarResult.innerHTML = '<p class="loading">加载中...</p>';
  
  try {
    const res = await fetch("/stock/similar", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ stock_id: stockId, top_k: 5 })
    });
    
    if (!res.ok) {
      throw new Error('获取相似股票失败');
    }
    
    const data = await res.json();
    state.currentSimilarStocks = data;
    
    if (data.similar_stocks && data.similar_stocks.length > 0) {
      let html = `<div class="card">
          <h3>相似股票推荐</h3>
          <p class="similar-title">基于您对 <strong>${data.similar_stocks[0].industry}</strong> 行业的兴趣</p>
          <div class="stock-list similar-stock-list">`;
      
      data.similar_stocks.forEach(stock => {
        const similarityColor = getSimilarityColor(stock.similarity);
        
        html += `
          <div class="card stock-card similar-stock" data-stock-id="${stock.stock_id}">
            <div class="stock-header">
              <div class="similarity-badge" style="background-color: ${similarityColor}">
                ${stock.similarity}% 相似
              </div>
              <strong>${stock.stock_name}</strong> <small>(${stock.ticker})</small>
              <span class="score">匹配分: ${stock.match_score.toFixed(2)}%</span>
            </div>
            <div class="stock-details">
              <p><strong>行业：</strong>${stock.industry}</p>
              <p><strong>预期收益：</strong>${stock.expected_return.toFixed(2)}%</p>
              <p><strong>风险评级：</strong>${getRiskLevelText(stock.risk_level)}</p>
            </div>
          </div>
        `;
      });
      
      html += `</div></div>`;
      similarResult.innerHTML = html;
      
      // 为相似股票添加点击事件
      document.querySelectorAll('.similar-stock').forEach(stock => {
        stock.addEventListener('click', function() {
          showStockExplanation(this.getAttribute('data-stock-id'));
        });
      });
      
    } else {
      similarResult.innerHTML = '<p class="no-data">暂无相似股票推荐</p>';
    }
  } catch (error) {
    console.error('获取相似股票失败:', error);
    similarResult.innerHTML = `<p class="error">获取相似股票失败，请重试</p>`;
  }
}

// 根据相似度获取颜色
function getSimilarityColor(similarity) {
  if (similarity >= 80) return '#4CAF50';
  if (similarity >= 60) return '#8BC34A';
  if (similarity >= 40) return '#FFC107';
  return '#FF9800';
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
  