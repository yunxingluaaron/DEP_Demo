import React, { useState, useEffect } from 'react';
import { AlertCircle, Play, RotateCcw, TrendingUp, X, Info, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';

const ABDecisionPredictor = () => {
  const [formData, setFormData] = useState({
    case_id: '',
    attainable_bottom: '',
    depth_reached: '',
    event_duration: '',
    obstruction_class: 'CEMENT_RELATED',
    deployment_method: 'Wireline',
    tools_used: [],
    confidence: 'medium'
  });
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [caseStarted, setCaseStarted] = useState(false);
  const [showExplanationModal, setShowExplanationModal] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [expandedSections, setExpandedSections] = useState({
    basic: true,
    tools: false
  });

  const obstructionClasses = [
    'CASING_ISSUES',
    'CEMENT_RELATED',
    'DEBRIS_AND_FOREIGN_OBJECTS',
    'EQUIPMENT_PROBLEMS',
    'FLUID_AND_FLOW_ISSUES',
    'FORMATION_AND_BOTTOM_ISSUES',
    'HOLE_INTEGRITY_AND_COLLAPSE'
  ];

  const deploymentMethods = [
    'Cable Tool',
    'Double Pole',
    'Dozer Rig',
    'Drilling Rig',
    'Power Swivel',
    'Pulling Rig',
    'Service Rig',
    'Single Pole',
    'Wireline'
  ];

  const allTools = [
    'CCL', 'Hydraulic Jars', 'Stem', 'Cable Tool Jars', 'GR', 'Mud pump',
    'Camera', 'Friction Socket', 'Circulation', 'Drilling Rig', 'Shooting',
    'Power Swivel', 'Mill', 'Bit', 'Magnet', 'Chisel Bit', 'Drill Pipe',
    'CBL', 'Spear', 'Center Spear', 'Mouse Trap Socket', 'Tri-cone',
    'Overshot', 'Grapple', 'Jet Cutter', 'Logging tool', 'Bailer',
    'Washover', 'Sand pump', 'Cable Tool', 'Chemical Cutter', 'Perforator'
  ];

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/health');
        const data = await response.json();
        setModelInfo(data);
      } catch (err) {
        console.error('Health check failed:', err);
      }
    };
    checkHealth();
  }, []);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleToolToggle = (tool) => {
    setFormData(prev => {
      const currentTools = prev.tools_used || [];
      return {
        ...prev,
        tools_used: currentTools.includes(tool)
          ? currentTools.filter(t => t !== tool)
          : [...currentTools, tool]
      };
    });
  };

  const startNewCase = async () => {
    if (!formData.case_id || !formData.attainable_bottom) {
      setError('Please provide Case ID and Target Depth');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:5000/api/start-case', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: formData.case_id,
          attainable_bottom: parseFloat(formData.attainable_bottom)
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setCaseStarted(true);
        setResult({ message: 'Case started successfully!' });
      } else {
        setError(data.error || 'Failed to start case');
      }
    } catch (err) {
      setError('Connection error. Make sure the backend server is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const addEventAndPredict = async () => {
    if (!formData.depth_reached || !formData.event_duration) {
      setError('Please provide Depth Reached and Event Duration');
      return;
    }

    const depthReached = parseFloat(formData.depth_reached);
    const targetDepth = parseFloat(formData.attainable_bottom);
    
    if (depthReached >= targetDepth) {
      setError('Depth reached or exceeded target. Operations should be concluded. No AB decision needed.');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:5000/api/add-event-predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: formData.case_id,
          depth_reached: parseFloat(formData.depth_reached),
          event_duration: parseFloat(formData.event_duration),
          obstruction_class: formData.obstruction_class,
          deployment_method: formData.deployment_method,
          tools_used: formData.tools_used || [],
          confidence: formData.confidence
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setResult(data.result);
      } else {
        setError(data.error || 'Failed to get prediction');
      }
    } catch (err) {
      setError('Connection error. Make sure the backend server is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      case_id: '',
      attainable_bottom: '',
      depth_reached: '',
      event_duration: '',
      obstruction_class: 'CEMENT_RELATED',
      deployment_method: 'Wireline',
      tools_used: [],
      confidence: 'medium'
    });
    setResult(null);
    setError('');
    setCaseStarted(false);
  };

  const renderExplanation = () => {
    if (!result || !result.recommendation) return null;

    const probability = result.hazard_rate || 0.5;
    const isAbandon = result.recommendation === 'ABANDON';
    const displayProbability = isAbandon ? probability : (1 - probability);
    
    return (
      <div className="bg-blue-50 p-4 rounded-md space-y-3">
        <div className="text-sm font-medium text-blue-800 mb-3">Model Analysis:</div>
        
        <div className="space-y-2 text-sm">
          <div className="bg-white p-3 rounded border border-blue-200">
            <div className="flex justify-between items-center mb-1">
              <span className="text-blue-700 font-medium">
                {isAbandon ? 'AB Probability:' : 'Continue Confidence:'}
              </span>
              <span className="text-blue-900 font-bold text-lg">{(displayProbability * 100).toFixed(1)}%</span>
            </div>
            <div className="text-xs text-gray-600 mb-2">
              Raw model output: P(AB) = {(probability * 100).toFixed(1)}%
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${isAbandon ? 'bg-green-600' : 'bg-red-600'}`}
                style={{ width: `${displayProbability * 100}%` }}
              ></div>
            </div>
          </div>

          <div className="text-xs text-blue-600 space-y-1">
            <div>• Current depth: {formData.depth_reached} ft / {formData.attainable_bottom} ft ({((parseFloat(formData.depth_reached) / parseFloat(formData.attainable_bottom)) * 100).toFixed(1)}%)</div>
            <div>• Obstruction type: {formData.obstruction_class.replace(/_/g, ' ')}</div>
            <div>• Deployment: {formData.deployment_method}</div>
            <div>• Event duration: {formData.event_duration} hours</div>
            <div>• Tools used: {formData.tools_used.length} tools</div>
            {result.step && <div>• Event sequence: #{result.step}</div>}
          </div>

          <div className="border-t pt-2 mt-3">
            <div className="flex justify-between items-center mb-2">
              <span className="text-blue-800 font-medium">Confidence:</span>
              <span className={`font-semibold ${result.confidence === 'HIGH' ? 'text-green-700' : 'text-yellow-700'}`}>
                {result.confidence}
              </span>
            </div>
            <button
              onClick={() => setShowExplanationModal(true)}
              className="w-full bg-blue-600 text-white py-1 px-3 rounded text-xs hover:bg-blue-700"
            >
              View Model Details
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-5xl mx-auto px-4">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-blue-600 mr-3" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">VineAI-AB Decision Predictor</h1>
                {modelInfo && (
                  <div className="text-xs text-gray-500 mt-1 flex items-center gap-2">
                    <span>Model: {modelInfo.model_type} - 55 Features</span>
                    {modelInfo.model_loaded ? (
                      <CheckCircle className="h-3 w-3 text-green-600" />
                    ) : (
                      <XCircle className="h-3 w-3 text-red-600" />
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md flex items-center">
              <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
              <span className="text-red-700">{error}</span>
            </div>
          )}

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Input Form */}
            <div className="lg:col-span-2 space-y-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Input Parameters</h2>
              
              {/* Case Setup */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-medium text-gray-700 mb-3">Case Setup</h3>
                <div className="grid md:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Case ID *</label>
                    <input
                      type="text"
                      name="case_id"
                      value={formData.case_id}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                      placeholder="e.g., Foxhill-1-2nd-attempt"
                      disabled={caseStarted}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Target Depth (ft) *</label>
                    <input
                      type="number"
                      name="attainable_bottom"
                      value={formData.attainable_bottom}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                      placeholder="e.g., 1916"
                      disabled={caseStarted}
                    />
                  </div>
                </div>
                
                {!caseStarted && (
                  <button
                    onClick={startNewCase}
                    disabled={loading}
                    className="w-full mt-3 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    {loading ? 'Starting...' : 'Start New Case'}
                  </button>
                )}
              </div>

              {/* Event Details */}
              {caseStarted && (
                <div className="space-y-4">
                  {/* Basic Parameters */}
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <button
                      onClick={() => toggleSection('basic')}
                      className="w-full flex items-center justify-between font-medium text-gray-700 mb-3"
                    >
                      <span>Event Parameters *</span>
                      {expandedSections.basic ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>
                    
                    {expandedSections.basic && (
                      <div className="grid md:grid-cols-2 gap-3">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Depth Reached (ft) *</label>
                          <input
                            type="number"
                            name="depth_reached"
                            value={formData.depth_reached}
                            onChange={handleInputChange}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            placeholder="e.g., 1017"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Event Duration (hours) *</label>
                          <input
                            type="number"
                            step="0.1"
                            name="event_duration"
                            value={formData.event_duration}
                            onChange={handleInputChange}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            placeholder="e.g., 168 (1 week)"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Obstruction Class *</label>
                          <select
                            name="obstruction_class"
                            value={formData.obstruction_class}
                            onChange={handleInputChange}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                          >
                            {obstructionClasses.map(cls => (
                              <option key={cls} value={cls}>
                                {cls.replace(/_/g, ' ')}
                              </option>
                            ))}
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Deployment Method</label>
                          <select
                            name="deployment_method"
                            value={formData.deployment_method}
                            onChange={handleInputChange}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                          >
                            {deploymentMethods.map(method => (
                              <option key={method} value={method}>{method}</option>
                            ))}
                          </select>
                        </div>
                        
                        <div className="md:col-span-2">
                          <label className="block text-sm font-medium text-gray-700 mb-1">Confidence Level</label>
                          <select
                            name="confidence"
                            value={formData.confidence}
                            onChange={handleInputChange}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                          >
                            <option value="low">Low - Uncertain conditions</option>
                            <option value="medium">Medium - Typical conditions</option>
                            <option value="high">High - Well-documented</option>
                          </select>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Tools Selection */}
                  <div className="bg-green-50 p-4 rounded-lg">
                    <button
                      onClick={() => toggleSection('tools')}
                      className="w-full flex items-center justify-between font-medium text-gray-700 mb-3"
                    >
                      <span>Tools Used ({formData.tools_used.length} selected)</span>
                      {expandedSections.tools ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>
                    
                    {expandedSections.tools && (
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-60 overflow-y-auto p-2">
                        {allTools.sort().map(tool => (
                          <label key={tool} className="flex items-center text-xs hover:bg-green-100 p-1 rounded cursor-pointer">
                            <input
                              type="checkbox"
                              checked={formData.tools_used.includes(tool)}
                              onChange={() => handleToolToggle(tool)}
                              className="mr-2"
                            />
                            <span>{tool}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </div>

                  <button
                    onClick={addEventAndPredict}
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                        Predicting...
                      </>
                    ) : (
                      'Add Event & Predict'
                    )}
                  </button>
                </div>
              )}

              <div className="flex gap-2">
                <button
                  onClick={resetForm}
                  className="flex-1 bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700 flex items-center justify-center text-sm"
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset Case
                </button>
                <button
                  onClick={() => setShowExplanationModal(true)}
                  className="flex-1 bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 text-sm flex items-center justify-center"
                >
                  <Info className="h-4 w-4 mr-2" />
                  Model Info
                </button>
              </div>
            </div>

            {/* Results Panel */}
            <div className="lg:col-span-1">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Prediction Results</h2>
              
              {result && (
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  {result.recommendation && (
                    <div className="space-y-3">
                      <div className={`p-4 rounded-md ${
                        result.recommendation === 'ABANDON' 
                          ? 'bg-green-100 border-2 border-green-300' 
                          : 'bg-red-100 border-2 border-red-300'
                      }`}>
                        <div className="flex items-start gap-3">
                          {result.recommendation === 'ABANDON' ? (
                            <CheckCircle className="h-6 w-6 text-green-700 mt-1 flex-shrink-0" />
                          ) : (
                            <XCircle className="h-6 w-6 text-red-700 mt-1 flex-shrink-0" />
                          )}
                          <div>
                            <div className="text-xs text-gray-600 font-normal mb-1">Decision:</div>
                            <div className={`font-bold text-lg ${result.recommendation === 'ABANDON' ? 'text-green-700' : 'text-red-700'}`}>
                              {result.recommendation === 'ABANDON' ? 'APPROVE AB' : 'REJECT AB'}
                            </div>
                            <div className="text-xs font-normal text-gray-600 mt-1">
                              {result.recommendation === 'ABANDON' ? 'Recommend stopping operations' : 'Recommend continuing operations'}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {renderExplanation()}
                    </div>
                  )}
                  
                  {result.message && !result.recommendation && (
                    <div className="text-green-700 font-medium text-center py-4">{result.message}</div>
                  )}
                </div>
              )}
              
              {!result && (
                <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <Info className="h-12 w-12 text-gray-400 mx-auto mb-3" />
                  <p className="text-gray-500 text-sm">
                    Complete the case setup and add an event to see prediction results
                  </p>
                </div>
              )}
            </div>
          </div>
          
          {/* Explanation Modal */}
          {showExplanationModal && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-lg max-w-3xl w-full max-h-[90vh] overflow-y-auto">
                <div className="flex items-center justify-between p-4 border-b sticky top-0 bg-white">
                  <h2 className="text-xl font-semibold text-gray-900">Model Information</h2>
                  <button
                    onClick={() => setShowExplanationModal(false)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="h-6 w-6" />
                  </button>
                </div>
                
                <div className="p-6 space-y-4 text-sm">
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Model Architecture</h3>
                    <p className="text-gray-700">
                      PyTorch neural network (3-layer fully connected: 56 → 128 → 64 → 1) trained on 
                      267 historical well events from California.
                    </p>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">55 Features Overview</h3>
                    <div className="bg-gray-50 p-3 rounded text-xs space-y-2">
                      <div>
                        <span className="font-medium">Core (2):</span> depth_reached, event_duration
                      </div>
                      <div>
                        <span className="font-medium">Obstruction Classes (7):</span> One-hot encoded obstruction types
                      </div>
                      <div>
                        <span className="font-medium">Deployment Methods (9):</span> One-hot encoded deployment methods
                      </div>
                      <div>
                        <span className="font-medium">Tools (32):</span> Binary indicators for each tool type
                      </div>
                      <div>
                        <span className="font-medium">Engineered (5):</span> confidence_num, depth_ratio, num_previous_events, 
                        cumulative_duration, tools_cum_unique
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">How It Works</h3>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 ml-2">
                      <li>Tracks case history event-by-event</li>
                      <li>Computes cumulative features automatically</li>
                      <li>Outputs P(AB) - probability of needing to abandon</li>
                      <li>Threshold: ≥50% = ABANDON, &lt;50% = CONTINUE</li>
                      <li>Confidence: HIGH if P(AB) &gt;70% or &lt;30%</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Performance Metrics</h3>
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <div className="font-semibold text-blue-900">91.8%</div>
                          <div className="text-xs text-blue-700">Training</div>
                        </div>
                        <div>
                          <div className="font-semibold text-blue-900">84.7%</div>
                          <div className="text-xs text-blue-700">Validation</div>
                        </div>
                        <div>
                          <div className="font-semibold text-blue-900">92.9%</div>
                          <div className="text-xs text-blue-700">Test</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-yellow-50 border border-yellow-200 p-3 rounded">
                    <h3 className="font-semibold text-yellow-900 mb-2 flex items-center gap-2">
                      <Info className="h-4 w-4" />
                      Important Notes
                    </h3>
                    <ul className="list-disc list-inside space-y-1 text-yellow-800 text-xs ml-2">
                      <li>Model trained on California well abandonment data</li>
                      <li>Simplified from 69 to 56 features for easier data collection</li>
                      <li>Best used as decision support, not sole decision maker</li>
                      <li>Accuracy may vary for novel situations not in training data</li>
                      <li>Consider consulting domain experts for critical decisions</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Key Decision Factors</h3>
                    <p className="text-gray-700">
                      The model primarily considers: depth progression over time, obstruction type patterns, 
                      tool usage diversity, cumulative operation duration, and historical progress trends.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ABDecisionPredictor;