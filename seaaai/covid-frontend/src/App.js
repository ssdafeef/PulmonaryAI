  import React, { useState } from 'react';
  import axios from 'axios';
  import { jsPDF } from 'jspdf';
  import {
    Upload,
    Activity,
    CheckCircle,
    Download,
    Clipboard,
    Trash2,
    History,
    Sparkles,
    ShieldCheck,
    FileText,
    Eye
  } from 'lucide-react';
  import './App.css';

  function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [heatmap, setHeatmap] = useState(null); // Added for Grad-CAM
    const [loading, setLoading] = useState(false);
    const [reportGeneratedAt, setReportGeneratedAt] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [history, setHistory] = useState([]);
    const [analysisMs, setAnalysisMs] = useState(null);
    const [copied, setCopied] = useState(false);
    const [llmInsight, setLlmInsight] = useState(null);
    const [llmLoading, setLlmLoading] = useState(false);
    const [llmError, setLlmError] = useState('');
    const [reportProfile, setReportProfile] = useState({
      hospitalName: 'Medical AI Diagnostic Portal',
      hospitalAddress: '123 Trichy, India',
      hospitalContact: '+91 (XXX) 123-4567',
      hospitalEmail: 'info@Medical_AI_Diagnostic.com',
      hospitalSite: 'www.Medical_AI_Diagnostic.com',
      doctorName: 'Dr. Olivia Greene',
      specialization: 'Doctor',
      visitDate: new Date().toISOString().slice(0, 10),
      patientName: 'Sarah Anderson',
      birthDate: '1989-01-01',
      medNumber: 'MA567891',
      ihi: '5556-9669-9654-7788',
      phone: '+1 (555) 789-0123',
      email: 's.anderson@mail.com'
    });

    // LLM/chat removed: state for chat assistant was here

    // Backend API base (local for development)
    const API_BASE_URL = (process.env.REACT_APP_API_URL || 'http://127.0.0.1:8010').replace(/\/$/, '');
    const API_URL = `${API_BASE_URL}/predict`;
    const LLM_API_URL = `${API_BASE_URL}/llm-insight`;
    // Chat integration removed

    const fileSizeLabel = file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : '-';
    const confidencePct = result ? (Number(result.confidence) * 100).toFixed(2) : '0.00';
    const riskLevel = result
      ? Number(result.confidence) >= 0.8
        ? 'High confidence'
        : Number(result.confidence) >= 0.55
          ? 'Medium confidence'
          : 'Low confidence'
      : 'Awaiting analysis';

    const getClinicalGuidance = (predictionValue) => {
      const normalized = String(predictionValue || '').toLowerCase();

      if (normalized.includes('covid')) {
        return {
          condition: 'COVID-19 pattern suspected',
          urgency: 'Prompt physician review recommended within the same day.',
          actions: [
            'Isolate patient as per local infection prevention protocol.',
            'Check oxygen saturation, respiratory rate, and temperature at triage.',
            'Correlate with symptoms and confirm using RT-PCR/rapid antigen as indicated.'
          ],
          tests: ['CBC and CRP', 'SPO2 monitoring', 'RT-PCR or antigen test'],
          warningSigns: ['SpO2 below 94%', 'Increasing shortness of breath', 'Persistent high fever or chest pain'],
          followUp: 'Reassess clinically in 12 to 24 hours or earlier if symptoms worsen.'
        };
      }

      if (normalized.includes('pneumonia')) {
        return {
          condition: 'Pneumonia pattern suspected',
          urgency: 'Medical assessment recommended within 24 hours.',
          actions: [
            'Perform full respiratory exam and vitals assessment.',
            'Consider bacterial versus viral etiology based on labs and history.',
            'Start management plan as advised by treating physician.'
          ],
          tests: ['CBC with differential', 'CRP/procalcitonin if available', 'Repeat chest imaging if clinically required'],
          warningSigns: ['Confusion', 'Hypotension', 'Rapidly worsening breathing pattern'],
          followUp: 'Follow-up review in 24 to 48 hours or sooner based on severity.'
        };
      }

      if (normalized.includes('tuberculosis') || normalized.includes('tb')) {
        return {
          condition: 'Tuberculosis-like pattern suspected',
          urgency: 'Early specialist review recommended and public health protocol should be considered.',
          actions: [
            'Apply respiratory precautions according to local guidelines.',
            'Evaluate prolonged cough history, weight loss, and fever/night sweats.',
            'Refer for microbiological confirmation and specialist evaluation.'
          ],
          tests: ['Sputum AFB/GeneXpert', 'ESR/CBC', 'Infectious disease or pulmonary consult'],
          warningSigns: ['Hemoptysis', 'Severe breathlessness', 'Clinical deterioration or dehydration'],
          followUp: 'Arrange documented follow-up and treatment linkage without delay.'
        };
      }

      if (normalized.includes('normal')) {
        return {
          condition: 'No major radiographic abnormality detected by AI',
          urgency: 'Routine clinical correlation advised.',
          actions: [
            'Correlate with symptoms and examination findings.',
            'If symptoms persist, clinician may still advise additional tests.',
            'Continue routine monitoring and preventive care.'
          ],
          tests: ['No immediate additional imaging unless clinically indicated'],
          warningSigns: ['New fever', 'Persistent cough', 'Drop in oxygen saturation'],
          followUp: 'Return if symptoms persist or worsen over the next few days.'
        };
      }

      return {
        condition: 'Abnormal pattern detected',
        urgency: 'Clinical review recommended soon for definitive diagnosis.',
        actions: [
          'Correlate with patient symptoms and physical exam.',
          'Review image quality and consider repeat imaging if needed.',
          'Seek physician interpretation and management planning.'
        ],
        tests: ['Laboratory and imaging workup per clinician judgment'],
        warningSigns: ['Worsening respiratory distress', 'Low oxygen saturation', 'Hemodynamic instability'],
        followUp: 'Schedule follow-up per physician recommendation.'
      };
    };

    const guidance = getClinicalGuidance(result?.prediction);
    const assessmentText = result
      ? `${reportProfile.patientName || 'The patient'} demonstrates ${guidance.condition.toLowerCase()} with an AI confidence score of ${confidencePct}%. Clinical presentation, examination, and laboratory context should be used to confirm this preliminary AI-supported finding.`
      : 'Run an analysis to generate assessment details.';
    const diagnosisText = result
      ? `Current AI-assisted diagnosis indicates ${result.prediction || 'N/A'}. This is a decision-support output and requires formal clinician validation before concluding a final diagnosis.`
      : 'Run an analysis to generate diagnosis details.';
    const prescriptionText = result
      ? `Initial management can follow physician-guided respiratory care protocol. Priority actions include: ${guidance.actions.slice(0, 2).join(' ')} Medication and treatment decisions must remain under licensed clinician supervision.`
      : 'Run an analysis to generate prescription guidance.';
    const llmData = llmInsight?.insight || null;
    const llmPatterns = Array.isArray(llmData?.patterns) ? llmData.patterns : [];
    const llmNarrative = llmData
      ? (llmData.narrative_paragraph
        || `${llmData.impression || ''} ${Array.isArray(llmData.evidence_points) ? `Key evidence includes ${llmData.evidence_points.slice(0, 3).join('; ')}.` : ''} ${Array.isArray(llmData.differentials) ? `Differential considerations include ${llmData.differentials.slice(0, 3).join('; ')}.` : ''} ${Array.isArray(llmData.action_plan) ? `Suggested immediate actions are ${llmData.action_plan.slice(0, 3).join('; ')}.` : ''}`.trim())
      : '';

    const onProfileChange = (field, value) => {
      setReportProfile((prev) => ({ ...prev, [field]: value }));
    };

    const fileToBase64 = (inputFile) => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const raw = String(reader.result || '');
        resolve(raw.includes(',') ? raw.split(',')[1] : raw);
      };
      reader.onerror = reject;
      reader.readAsDataURL(inputFile);
    });

    const generateAnnotatedImage = (imageSrc, patterns = []) => new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        try {
          const MAX_W = 1200;
          const iw = img.naturalWidth || img.width;
          const ih = img.naturalHeight || img.height;
          const scale = Math.min(1, MAX_W / iw);
          const cw = Math.max(200, Math.round(iw * scale));
          const ch = Math.max(200, Math.round(ih * scale));
          const canvas = document.createElement('canvas');
          canvas.width = cw;
          canvas.height = ch;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, cw, ch);
          ctx.drawImage(img, 0, 0, cw, ch);

          // draw pattern boxes
          ctx.lineWidth = Math.max(2, Math.round(Math.min(cw, ch) * 0.004));
          ctx.font = `${Math.max(10, Math.round(cw * 0.03))}px sans-serif`;
          patterns.forEach((p) => {
            const r = p.region || {};
            const x = (Number(r.x || 0)) * cw;
            const y = (Number(r.y || 0)) * ch;
            const w = (Number(r.w || 0.2)) * cw;
            const h = (Number(r.h || 0.2)) * ch;
            ctx.strokeStyle = 'rgba(15,118,110,0.95)';
            ctx.fillStyle = 'rgba(15,118,110,0.12)';
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
            ctx.fillStyle = 'rgba(6,63,58,0.95)';
            const label = `${p.name || 'Pattern'} ${p.confidence ? `(${Math.round(p.confidence * 100)}%)` : ''}`;
            const textX = x + 6;
            const textY = y + Math.max(14, Math.round(ctx.font.match(/\d+/) ? Number(ctx.font.match(/\d+/)[0]) : 12));
            ctx.fillText(label, textX, textY);
          });

          const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
          resolve(dataUrl);
        } catch (err) {
          reject(err);
        }
      };
      img.onerror = reject;
      img.src = imageSrc;
    });

    const setSelectedFile = (selectedFile) => {
      if (!selectedFile) return;
      if (preview) URL.revokeObjectURL(preview);

      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setHeatmap(null);
      setReportGeneratedAt(null);
      setAnalysisMs(null);
      setCopied(false);
      setLlmInsight(null);
      setLlmError('');
    };

    const onFileChange = (e) => setSelectedFile(e.target.files[0]);
    const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
    const onDragLeave = () => setIsDragging(false);
    const onDrop = (e) => {
      e.preventDefault();
      setIsDragging(false);
      setSelectedFile(e.dataTransfer.files?.[0]);
    };

    const handlePredict = async () => {
      if (!file) return;
      setLoading(true);
      setCopied(false);

      const formData = new FormData();
      formData.append('file', file);
      const start = performance.now();

      try {
        const response = await axios.post(API_URL, formData);
        const generatedAt = new Date();
        const duration = performance.now() - start;
        const predictionResult = response.data || {};
        const confidenceValue = Number(predictionResult.confidence);

        if (predictionResult.error || !Number.isFinite(confidenceValue)) {
          throw new Error(predictionResult.error || 'Prediction response did not include a valid confidence score.');
        }

        setResult(predictionResult);
        setHeatmap(predictionResult.heatmap); // Capture the heatmap from API
        setReportGeneratedAt(generatedAt);
        setAnalysisMs(duration);
        setLlmInsight(null);
        setLlmError('');

        setHistory((prev) => {
          const next = [
            {
              id: `${generatedAt.getTime()}-${Math.random().toString(16).slice(2)}`,
              prediction: predictionResult?.prediction || 'N/A',
              confidence: confidenceValue,
              createdAt: generatedAt,
              fileName: file.name
            },
            ...prev
          ];
          return next.slice(0, 6);
        });
      } catch (err) {
        console.error(err);
        alert(err?.message || 'Connection failed. Ensure prediction API is reachable and backend is running.');
      } finally {
        setLoading(false);
      }
    };

    const handleGenerateInsight = async () => {
      if (!result) return;
      setLlmLoading(true);
      setLlmError('');

      try {
        const imageBase64 = file ? await fileToBase64(file) : '';
        const response = await axios.post(LLM_API_URL, {
          prediction: result?.prediction,
          confidence: Number(result?.confidence || 0),
          patientName: reportProfile.patientName,
          hospitalName: reportProfile.hospitalName,
          doctorName: reportProfile.doctorName,
          heatmapAvailable: Boolean(heatmap),
          imageBase64,
          imageMimeType: file?.type || 'image/jpeg'
        });
        setLlmInsight(response.data || null);
      } catch (err) {
        console.error(err);
        const backendMessage = err?.response?.data?.warning || err?.response?.data?.error || err?.response?.data?.message;
        if (backendMessage) {
          setLlmError(`Insight request failed: ${backendMessage}`);
        } else if (err?.code === 'ERR_NETWORK') {
          setLlmError(`Cannot reach insight API at ${LLM_API_URL}. Ensure backend is running on port 8000.`);
        } else {
          setLlmError(`Could not generate AI insight right now (${err?.message || 'unknown error'}).`);
        }
      } finally {
        setLlmLoading(false);
      }
    };

    const handleDownloadReport = () => {
      if (!result) return;
      const now = reportGeneratedAt || new Date();
      const reportId = `RAD-${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
      const reportContent = [
        reportProfile.hospitalName,
        reportProfile.hospitalAddress,
        `Contact: ${reportProfile.hospitalContact} | ${reportProfile.hospitalEmail}`,
        reportProfile.hospitalSite,
        '',
        'MEDICAL REPORT',
        '============================',
        `Report ID: ${reportId}`,
        `Generated At: ${now.toLocaleString()}`,
        '',
        'VISIT INFO',
        '----------',
        `Doctor Name: ${reportProfile.doctorName}`,
        `Specialization: ${reportProfile.specialization}`,
        `Visit Date: ${reportProfile.visitDate}`,
        '',
        'PATIENT INFO',
        '------------',
        `Full Name: ${reportProfile.patientName}`,
        `Birth Date: ${reportProfile.birthDate}`,
        `Medical Number: ${reportProfile.medNumber}`,
        `IHI: ${reportProfile.ihi}`,
        `Phone: ${reportProfile.phone}`,
        `Email: ${reportProfile.email}`,
        '',
        `Original Image: ${file?.name || 'Unknown file'}`,
        `Image Size: ${fileSizeLabel}`,
        `Inference Latency: ${analysisMs ? `${Math.round(analysisMs)} ms` : 'N/A'}`,
        '',
        'AI FINDINGS',
        '-----------',
        `Classification: ${result.prediction || 'N/A'}`,
        `AI Confidence: ${confidencePct}%`,
        `Confidence Band: ${riskLevel}`,
        `Grad-CAM Heatmap Included In UI: ${heatmap ? 'Yes' : 'No'}`,
        '',
        'CLINICAL INTERPRETATION',
        '-----------------------',
        `Likely Condition: ${guidance.condition}`,
        `Urgency Note: ${guidance.urgency}`,
        '',
        'RECOMMENDED NEXT STEPS',
        '----------------------',
        ...guidance.actions.map((item, index) => `${index + 1}. ${item}`),
        '',
        'SUGGESTED ADDITIONAL TESTS',
        '--------------------------',
        ...guidance.tests.map((item, index) => `${index + 1}. ${item}`),
        '',
        'RED FLAG SYMPTOMS',
        '-----------------',
        ...guidance.warningSigns.map((item, index) => `${index + 1}. ${item}`),
        '',
        'ASSESSMENT',
        '----------',
        assessmentText,
        '',
        'DIAGNOSIS',
        '---------',
        diagnosisText,
        '',
        'PRESCRIPTION / PLAN',
        '-------------------',
        prescriptionText,
        '',
        ...(llmData ? [
          'LLM CLINICAL INSIGHT',
          '--------------------',
          `Model: ${llmInsight?.model || 'N/A'}`,
          `Source: ${llmInsight?.source || 'remote-llm'}`,
          `Latency: ${llmInsight?.latency_ms ?? 'N/A'} ms`,
          `Narrative Summary: ${llmNarrative || 'N/A'}`,
          'Evidence Points:',
          ...(Array.isArray(llmData.evidence_points) ? llmData.evidence_points.map((item, index) => `${index + 1}. ${item}`) : ['1. N/A']),
          'Differentials:',
          ...(Array.isArray(llmData.differentials) ? llmData.differentials.map((item, index) => `${index + 1}. ${item}`) : ['1. N/A']),
          'Action Plan:',
          ...(Array.isArray(llmData.action_plan) ? llmData.action_plan.map((item, index) => `${index + 1}. ${item}`) : ['1. N/A']),
          'Pattern Findings (Image-Based):',
          ...(llmPatterns.length
            ? llmPatterns.map((item, index) => {
              const region = item?.region || {};
              return `${index + 1}. ${item?.name || 'Pattern'} (${item?.confidence || 'N/A'}) - ${item?.finding || 'N/A'} [x:${region?.x ?? 'N/A'}, y:${region?.y ?? 'N/A'}, w:${region?.w ?? 'N/A'}, h:${region?.h ?? 'N/A'}]`;
            })
            : ['1. No explicit pattern map returned by model.']),
          `Caution: ${llmData.caution || 'N/A'}`,
          `Uncertainty: ${llmData.uncertainty || 'N/A'}`,
          ''
        ] : []),
        `Follow-Up Advice: ${guidance.followUp}`,
        '',
        'Disclaimer: This AI output is decision-support only for research/demo use and must be reviewed by a licensed clinician before any diagnosis or treatment.'
      ].join('\n');

      const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `report-${now.getTime()}.txt`;
      link.click();
    };

    const handleDownloadPdfReport = async () => {
      if (!result) return;
      const now = reportGeneratedAt || new Date();
      const reportId = `RAD-${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
      const doc = new jsPDF();
      let y = 14;

      const ensureSpace = (needed = 8) => {
        if (y + needed > 276) {
          doc.addPage();
          y = 14;
        }
      };

      const addSeparator = () => {
        ensureSpace(6);
        doc.setDrawColor(190, 190, 190);
        doc.line(14, y, 196, y);
        y += 5;
      };

      const addHeading = (text) => {
        ensureSpace(10);
        doc.setTextColor(21, 105, 85);
        doc.setFont('times', 'bold');
        doc.setFontSize(13);
        doc.text(text, 14, y);
        y += 7;
      };

      const addCenter = (text, size = 11, style = 'normal', color = [0, 0, 0]) => {
        ensureSpace(8);
        doc.setTextColor(color[0], color[1], color[2]);
        doc.setFont('times', style);
        doc.setFontSize(size);
        doc.text(text, 105, y, { align: 'center' });
        y += 6;
      };

      const addLine = (text) => {
        ensureSpace(8);
        doc.setTextColor(32, 32, 32);
        doc.setFont('times', 'normal');
        doc.setFontSize(10.5);
        const wrapped = doc.splitTextToSize(text, 182);
        doc.text(wrapped, 14, y);
        y += wrapped.length * 5 + 1;
      };

      const addPairLine = (leftLabel, leftValue, rightLabel, rightValue) => {
        ensureSpace(8);
        doc.setTextColor(60, 60, 60);
        doc.setFont('times', 'bold');
        doc.setFontSize(10.2);
        doc.text(`${leftLabel}:`, 14, y);
        doc.setFont('times', 'normal');
        doc.text(String(leftValue || '-'), 46, y);

        doc.setFont('times', 'bold');
        doc.text(`${rightLabel}:`, 112, y);
        doc.setFont('times', 'normal');
        doc.text(String(rightValue || '-'), 146, y);
        y += 6;
      };

      const addList = (items) => {
        items.forEach((item, index) => addLine(`${index + 1}. ${item}`));
      };

      addCenter(reportProfile.hospitalName, 18, 'bold', [30, 120, 95]);
      addCenter(reportProfile.hospitalAddress, 10, 'normal', [60, 60, 60]);
      addSeparator();
      addCenter('MEDICAL REPORT', 17, 'bold', [15, 15, 15]);
      addSeparator();
      addLine(`Report ID: ${reportId}`);
      addLine(`Report ID: ${reportId}`);
      addLine(`Generated At: ${now.toLocaleString()}`);
      addHeading('Visit Info');
      addPairLine('Doctor', reportProfile.doctorName, 'Visit Date', reportProfile.visitDate);
      addPairLine('Specialization', reportProfile.specialization, 'Report Type', 'AI Radiology Support');

      addHeading('Patient Info');
      addPairLine('Full Name', reportProfile.patientName, 'Birth Date', reportProfile.birthDate);
      addPairLine('Med Number', reportProfile.medNumber, 'IHI', reportProfile.ihi);
      addPairLine('Phone', reportProfile.phone, 'Email', reportProfile.email);

      addHeading('Image and Analysis Metadata');
      addPairLine('Image', file?.name || 'Unknown file', 'Image Size', fileSizeLabel);
      addPairLine('Latency', analysisMs ? `${Math.round(analysisMs)} ms` : 'N/A', 'Heatmap', heatmap ? 'Available' : 'Not available');

      y += 2;
      addHeading('AI Findings');
      addLine(`Classification: ${result.prediction || 'N/A'}`);
      addLine(`AI Confidence: ${confidencePct}%`);
      addLine(`Confidence Band: ${riskLevel}`);
      addLine(`Grad-CAM Heatmap Included In UI: ${heatmap ? 'Yes' : 'No'}`);

      y += 2;
      addHeading('Clinical Interpretation');
      addLine(`Likely Condition: ${guidance.condition}`);
      addLine(`Urgency Note: ${guidance.urgency}`);

      y += 2;
      addHeading('Recommended Next Steps');
      addList(guidance.actions);

      y += 2;
      addHeading('Suggested Additional Tests');
      addList(guidance.tests);

      y += 2;
      addHeading('Red Flag Symptoms');
      addList(guidance.warningSigns);

      y += 2;
      addHeading('Follow-Up');
      addLine(guidance.followUp);

      y += 2;
      addHeading('Assessment');
      addLine(assessmentText);

      y += 2;
      addHeading('Diagnosis');
      addLine(diagnosisText);

      y += 2;
      addHeading('Prescription / Plan');
      addLine(prescriptionText);

      if (llmData) {
        y += 2;
        addHeading('LLM Clinical Insight');
        addLine(`Model: ${llmInsight?.model || 'N/A'} | Source: ${llmInsight?.source || 'remote-llm'} | Latency: ${llmInsight?.latency_ms ?? 'N/A'} ms`);
        addLine(`Narrative Summary: ${llmNarrative || 'N/A'}`);

        y += 2;
        addHeading('LLM Evidence Points');
        addList(Array.isArray(llmData.evidence_points) ? llmData.evidence_points : ['N/A']);

        y += 2;
        addHeading('LLM Differentials');
        addList(Array.isArray(llmData.differentials) ? llmData.differentials : ['N/A']);

        y += 2;
        addHeading('LLM Action Plan');
        addList(Array.isArray(llmData.action_plan) ? llmData.action_plan : ['N/A']);

        y += 2;
        addHeading('LLM Pattern Findings');
        addList(
          llmPatterns.length
            ? llmPatterns.map((item) => {
              const region = item?.region || {};
              return `${item?.name || 'Pattern'} (${item?.confidence || 'N/A'}) - ${item?.finding || 'N/A'} [x:${region?.x ?? 'N/A'}, y:${region?.y ?? 'N/A'}, w:${region?.w ?? 'N/A'}, h:${region?.h ?? 'N/A'}]`;
            })
            : ['No explicit pattern map returned by model.']
        );

        addLine(`Caution: ${llmData.caution || 'N/A'}`);
        addLine(`Uncertainty: ${llmData.uncertainty || 'N/A'}`);
      }

      // If the LLM returned pattern annotations and we have the preview image, render an annotated image and embed in the PDF
      if (llmPatterns.length && preview) {
        try {
          // create annotated image from preview + patterns
          const annotatedDataUrl = await generateAnnotatedImage(preview, llmPatterns);
          // Add a new page for the annotated image
          doc.addPage();
          y = 14;
          addHeading('LLM Annotated Image');
          const imgProps = doc.getImageProperties(annotatedDataUrl);
          const imgTargetW = 182; // page printable width margin
          const imgTargetH = (imgProps.height * imgTargetW) / imgProps.width;
          ensureSpace(imgTargetH + 6);
          doc.addImage(annotatedDataUrl, 'JPEG', 14, y, imgTargetW, imgTargetH);
          y += imgTargetH + 6;
        } catch (err) {
          console.error('Failed to add annotated image to PDF', err);
        }
      }

      y += 2;
      addHeading('Disclaimer');
      addLine('This AI output is decision-support only for research/demo use and must be reviewed by a licensed clinician before any diagnosis or treatment.');

      ensureSpace(14);
      y += 4;
      doc.setTextColor(150, 150, 150);
      doc.setFont('times', 'normal');
      doc.setFontSize(9.5);
      doc.text(
        `For inquiries and appointments: ${reportProfile.hospitalContact} | ${reportProfile.hospitalEmail}`,
        105,
        y,
        { align: 'center' }
      );
      y += 5;
      doc.text(reportProfile.hospitalSite, 105, y, { align: 'center' });

      doc.save(`report-${Date.now()}.pdf`);
    };

    const handleCopySummary = async () => {
      if (!result) return;
      const summary = `Diagnosis: ${result.prediction} | Confidence: ${confidencePct}% | File: ${file?.name}`;
      await navigator.clipboard.writeText(summary);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    };

    const handleClearSession = () => {
      if (preview) URL.revokeObjectURL(preview);
      setFile(null);
      setPreview(null);
      setResult(null);
      setHeatmap(null);
      setHistory([]);
      setLlmInsight(null);
      setLlmError('');
      // chat state cleared (removed)
    };

    // Chat/send functions removed because LLM integration is disabled

    return (
      <div className="app-shell">
        <div className="bg-orb bg-orb-one" />
        <div className="bg-orb bg-orb-two" />

        <header className="hero">
          <div className="hero-title-wrap">
            <Activity size={32} />
            <h1>Medical AI Diagnostic Portal</h1>
          </div>
          <p className="hero-subtitle">Agentic AI integration for hospital resource optimization and triage support.</p>
          
          <div className="hero-metrics">
            <div className="metric-card">
              <Sparkles size={18} />
              <div><p>Analyses</p><strong>{history.length}</strong></div>
            </div>
            <div className="metric-card">
              <ShieldCheck size={18} />
              <div><p>Status</p><strong>{loading ? 'Running' : 'Ready'}</strong></div>
            </div>
            <div className="metric-card">
              <FileText size={18} />
              <div><p>Size</p><strong>{fileSizeLabel}</strong></div>
            </div>
          </div>
        </header>

        <main className="dashboard-grid">
          <section className="panel upload-panel">
            <div className={`upload-area ${isDragging ? 'drag-active' : ''}`} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
              {preview ? <img src={preview} alt="Preview" className="preview" /> : (
                <div className="placeholder">
                  <Upload size={48} /><p>Drop X-ray here or browse files</p>
                </div>
              )}
              <input type="file" accept="image/*" onChange={onFileChange} className="hidden-input" id="xray-upload" />
              <label htmlFor="xray-upload" className="upload-label">{file ? "Change Image" : "Choose File"}</label>
            </div>

            <button onClick={handlePredict} disabled={!file || loading} className="primary-button">
              {loading ? "Analyzing..." : "Run AI Prediction"}
            </button>

            <div className="quick-actions">
              <button onClick={handleCopySummary} disabled={!result} className="ghost-button">
                <Clipboard size={16} /> {copied ? 'Copied' : 'Copy Summary'}
              </button>
              <button onClick={handleClearSession} className="ghost-button danger-ghost">
                <Trash2 size={16} /> Clear Session
              </button>
            </div>

            <div className="report-form-card">
              <h3 className="report-form-title">Report Profile</h3>
              <p className="report-form-subtitle">These details are used in TXT and PDF report exports.</p>
              <div className="report-form-grid">
                <label>
                  Hospital Name
                  <input value={reportProfile.hospitalName} onChange={(e) => onProfileChange('hospitalName', e.target.value)} />
                </label>
                <label>
                  Hospital Address
                  <input value={reportProfile.hospitalAddress} onChange={(e) => onProfileChange('hospitalAddress', e.target.value)} />
                </label>
                <label>
                  Doctor Name
                  <input value={reportProfile.doctorName} onChange={(e) => onProfileChange('doctorName', e.target.value)} />
                </label>
                <label>
                  Specialization
                  <input value={reportProfile.specialization} onChange={(e) => onProfileChange('specialization', e.target.value)} />
                </label>
                <label>
                  Visit Date
                  <input type="date" value={reportProfile.visitDate} onChange={(e) => onProfileChange('visitDate', e.target.value)} />
                </label>
                <label>
                  Patient Name
                  <input value={reportProfile.patientName} onChange={(e) => onProfileChange('patientName', e.target.value)} />
                </label>
                <label>
                  Birth Date
                  <input type="date" value={reportProfile.birthDate} onChange={(e) => onProfileChange('birthDate', e.target.value)} />
                </label>
                <label>
                  Medical Number
                  <input value={reportProfile.medNumber} onChange={(e) => onProfileChange('medNumber', e.target.value)} />
                </label>
                <label>
                  IHI
                  <input value={reportProfile.ihi} onChange={(e) => onProfileChange('ihi', e.target.value)} />
                </label>
                <label>
                  Phone
                  <input value={reportProfile.phone} onChange={(e) => onProfileChange('phone', e.target.value)} />
                </label>
                <label>
                  Email
                  <input value={reportProfile.email} onChange={(e) => onProfileChange('email', e.target.value)} />
                </label>
                <label>
                  Hospital Contact
                  <input value={reportProfile.hospitalContact} onChange={(e) => onProfileChange('hospitalContact', e.target.value)} />
                </label>
                <label>
                  Hospital Email
                  <input value={reportProfile.hospitalEmail} onChange={(e) => onProfileChange('hospitalEmail', e.target.value)} />
                </label>
                <label>
                  Hospital Site
                  <input value={reportProfile.hospitalSite} onChange={(e) => onProfileChange('hospitalSite', e.target.value)} />
                </label>
              </div>
            </div>
          </section>

          {result && (
            <section className="panel result-panel">
              <div className="result-header">
                <div className="result-title"><CheckCircle size={24} /><h3>Diagnosis Results</h3></div>
                <span className={`risk-pill ${Number(result.confidence) >= 0.8 ? 'risk-high' : 'risk-low'}`}>
                  {riskLevel}
                </span>
              </div>

              {/* Grad-CAM Heatmap Section */}
              {heatmap && (
                <div className="heatmap-box" style={{ margin: '15px 0', padding: '10px', backgroundColor: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', color: '#64748b', fontSize: '12px' }}>
                    <Eye size={14} /> <span>AI Interpretability Layer (Grad-CAM)</span>
                  </div>
                  <img 
                    src={`data:image/jpeg;base64,${heatmap}`} 
                    alt="AI Heatmap" 
                    style={{ width: '100%', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }} 
                  />
                </div>
              )}

              <div className="result-body">
                <div className="data-row"><span>Classification:</span><span className="badge">{result.prediction}</span></div>
                <div className="data-row"><span>AI Confidence:</span><span className="strong-text">{confidencePct}%</span></div>
                <div className="confidence-track"><div className="confidence-fill" style={{ width: `${confidencePct}%` }} /></div>
              </div>

              <div className="guidance-box">
                <h4 className="guidance-title">What To Do Next</h4>
                <p className="guidance-summary"><strong>Likely Condition:</strong> {guidance.condition}</p>
                <p className="guidance-summary"><strong>Urgency:</strong> {guidance.urgency}</p>
                <div className="guidance-grid">
                  <div>
                    <p className="guidance-label">Recommended Actions</p>
                    <ul className="guidance-list">
                      {guidance.actions.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                  </div>
                  <div>
                    <p className="guidance-label">Red Flags</p>
                    <ul className="guidance-list warning-list">
                      {guidance.warningSigns.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                  </div>
                </div>
                <p className="guidance-followup"><strong>Follow-Up:</strong> {guidance.followUp}</p>
              </div>

              <div className="llm-insight-box">
                <div className="llm-insight-header">
                  <div className="llm-insight-title">
                    <Sparkles size={16} />
                    <h4>AI Clinical Insight</h4>
                    {llmInsight?.model && <span className="llm-model-badge">{llmInsight.model}</span>}
                    <span className={`llm-annotated-badge ${llmPatterns.length ? 'has' : 'no'}`}>
                      {llmPatterns.length ? `Annotated Image (${llmPatterns.length})` : 'No Annotations'}
                    </span>
                  </div>
                  <button
                    onClick={handleGenerateInsight}
                    disabled={llmLoading}
                    className="download-button-secondary llm-generate-btn"
                  >
                    {llmLoading ? 'Generating...' : 'Generate Insight'}
                  </button>
                </div>

                {llmLoading && <p className="llm-status">Synthesizing reasoning and action plan...</p>}
                {llmError && <p className="llm-error">{llmError}</p>}

                {llmInsight?.insight && (
                  <div className="llm-insight-content">
                    <p className="llm-meta">
                      Source: {llmInsight.source || 'remote-llm'} | Latency: {llmInsight.latency_ms ?? 'N/A'} ms | Gemini-backed LLM
                    </p>
                    <p><strong>Impression:</strong> {llmInsight.insight.impression}</p>
                    {llmNarrative && <p><strong>Narrative Summary:</strong> {llmNarrative}</p>}

                    {preview && llmPatterns.length > 0 && (
                      <div className="pattern-map-box">
                        <p className="guidance-label">LLM Pattern Map</p>
                        <div className="pattern-map-stage">
                          <img src={preview} alt="LLM Pattern Overlay" className="pattern-map-image" />
                          {llmPatterns.map((item, index) => {
                            const region = item?.region || {};
                            const x = Math.max(0, Math.min(1, Number(region.x ?? 0))) * 100;
                            const y = Math.max(0, Math.min(1, Number(region.y ?? 0))) * 100;
                            const w = Math.max(0, Math.min(1, Number(region.w ?? 0.2))) * 100;
                            const h = Math.max(0, Math.min(1, Number(region.h ?? 0.2))) * 100;
                            return (
                              <div
                                key={`${item?.name || 'pattern'}-${index}`}
                                className="pattern-box"
                                style={{ left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` }}
                              >
                                <span>{item?.name || 'Pattern'}</span>
                              </div>
                            );
                          })}
                        </div>
                        <ul className="guidance-list">
                          {llmPatterns.map((item, index) => (
                            <li key={`pattern-${index}`}>
                              <strong>{item?.name || 'Pattern'}:</strong> {item?.finding || 'No details'} ({item?.confidence || 'N/A'})
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="llm-grid">
                      <div>
                        <p className="guidance-label">Evidence Points</p>
                        <ul className="guidance-list">
                          {(llmInsight.insight.evidence_points || []).map((item) => <li key={item}>{item}</li>)}
                        </ul>
                      </div>
                      <div>
                        <p className="guidance-label">Differentials</p>
                        <ul className="guidance-list">
                          {(llmInsight.insight.differentials || []).map((item) => <li key={item}>{item}</li>)}
                        </ul>
                      </div>
                    </div>

                    <p className="guidance-label">Action Plan</p>
                    <ul className="guidance-list">
                      {(llmInsight.insight.action_plan || []).map((item) => <li key={item}>{item}</li>)}
                    </ul>

                    <p className="llm-caution">
                      <strong>Caution:</strong> {llmInsight.insight.caution} <strong>Uncertainty:</strong>{' '}
                      {llmInsight.insight.uncertainty || 'N/A'}
                    </p>
                  </div>
                )}
              </div>

              <div className="download-actions">
                <button onClick={handleDownloadReport} className="download-button"><Download size={16} /> TXT</button>
                <button onClick={handleDownloadPdfReport} className="download-button-secondary"><Download size={16} /> PDF</button>
              </div>
              <p className="disclaimer">Research use only. Not for final clinical diagnostic use.</p>
            </section>
          )}

          <section className="panel history-panel">
            <div className="history-header"><History size={20} /><h3>Recent Analyses</h3></div>
            {history.length === 0 ? <p className="history-empty">No analyses yet.</p> : (
              <ul className="history-list">
                {history.map((item) => (
                  <li key={item.id} className="history-item">
                    <div><p className="history-title">{item.prediction}</p><p className="history-meta">{item.fileName}</p></div>
                    <div className="history-right"><span>{(item.confidence * 100).toFixed(1)}%</span><small>{new Date(item.createdAt).toLocaleTimeString()}</small></div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* Chat panel removed */}
        </main>
      </div>
    );
  }

  export default App;
