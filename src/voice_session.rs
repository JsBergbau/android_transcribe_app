use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jni::JNIEnv;
use jni::objects::{GlobalRef, JObject};
use transcribe_rs::TranscriptionEngine;

use crate::{assets, engine};

/// Shared flag for model loading across IME + RecognizeActivity
static MODEL_LOADING: AtomicBool = AtomicBool::new(false);

pub struct SendStream(#[allow(dead_code)] pub cpal::Stream);
unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

pub struct VoiceSessionState {
    pub stream: Option<SendStream>,
    pub audio_buffer: Arc<Mutex<Vec<f32>>>,
    pub jvm: Arc<jni::JavaVM>,
    pub target_ref: GlobalRef,
    pub last_level_sent: Arc<Mutex<std::time::Instant>>,
}

fn notify_status(env: &mut JNIEnv, obj: &JObject, msg: &str) {
    if let Ok(jmsg) = env.new_string(msg) {
        let _ = env.call_method(obj, "onStatusUpdate", "(Ljava/lang/String;)V", &[(&jmsg).into()]);
    }
}

fn notify_level(env: &mut JNIEnv, obj: &JObject, level: f32) {
    let _ = env.call_method(obj, "onAudioLevel", "(F)V", &[level.into()]);
}


fn notify_text(env: &mut JNIEnv, obj: &JObject, text: &str) {
    if let Ok(jtxt) = env.new_string(text) {
        let _ = env.call_method(obj, "onTextTranscribed", "(Ljava/lang/String;)V", &[(&jtxt).into()]);
    }
}

pub fn init_session(env: JNIEnv, target: JObject) -> VoiceSessionState {
    android_logger::init_once(android_logger::Config::default().with_max_level(log::LevelFilter::Info));

    let vm = env.get_java_vm().expect("Failed to get JavaVM");
    let vm_arc = Arc::new(vm);
    let target_ref = env.new_global_ref(&target).expect("Failed to ref target");

    let state = VoiceSessionState {
        stream: None,
        audio_buffer: Arc::new(Mutex::new(Vec::new())),
        jvm: vm_arc.clone(),
        target_ref: target_ref.clone(),
        last_level_sent: Arc::new(Mutex::new(std::time::Instant::now())),
    };

    // Ensure engine loaded (shared for both entry points)
    let vm_clone = vm_arc.clone();
    let target_ref_clone = target_ref.clone();

    std::thread::spawn(move || {
        if !engine::is_engine_loaded() {
            MODEL_LOADING.store(true, Ordering::Release);

            if let Ok(mut env) = vm_clone.attach_current_thread() {
                let obj = target_ref_clone.as_obj();
                notify_status(&mut env, obj, "Loading model...");

                match assets::extract_assets(&mut env, obj) {
                    Ok(path) => {
                        let mut eng = transcribe_rs::engines::parakeet::ParakeetEngine::new();
                        match eng.load_model_with_params(
                            &path,
                            transcribe_rs::engines::parakeet::ParakeetModelParams::int8()
                        ) {
                            Ok(_) => {
                                engine::set_engine(eng);
                                MODEL_LOADING.store(false, Ordering::Release);
                                notify_status(&mut env, obj, "Ready");
                            }
                            Err(e) => {
                                MODEL_LOADING.store(false, Ordering::Release);
                                notify_status(&mut env, obj, &format!("Error: {}", e));
                            }
                        }
                    }
                    Err(e) => {
                        MODEL_LOADING.store(false, Ordering::Release);
                        notify_status(&mut env, obj, &format!("Error: {}", e));
                    }
                }
            } else {
                MODEL_LOADING.store(false, Ordering::Release);
            }
        } else {
            if let Ok(mut env) = vm_clone.attach_current_thread() {
                notify_status(&mut env, target_ref_clone.as_obj(), "Ready");
            }
        }
    });

    state
}

pub fn start_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            notify_status(&mut env, state.target_ref.as_obj(), "Error: no input device");
            return;
        }
    };

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    state.audio_buffer.lock().unwrap().clear();
    let buffer_clone = state.audio_buffer.clone();
        
    let jvm = state.jvm.clone();
    let target_ref = state.target_ref.clone();
    let last_sent = state.last_level_sent.clone();    

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            buffer_clone.lock().unwrap().extend_from_slice(data);
			
			// compute RMS
			let mut sum = 0.0f32;
			for &x in data {
				sum += x * x;
			}
			let rms = (sum / (data.len() as f32)).sqrt();
			let level = (rms * 6.0).clamp(0.0, 1.0);

			// throttle updates
			let mut last = last_sent.lock().unwrap();
			if last.elapsed() >= std::time::Duration::from_millis(50) {
				*last = std::time::Instant::now();

				if let Ok(mut env) = jvm.attach_current_thread() {
					let obj = target_ref.as_obj();
					notify_level(&mut env, obj, level);
				}
			}            
        },
        |e| log::error!("Stream err: {}", e),
        None,
    );

    match stream {
        Ok(s) => {
            s.play().ok();
            state.stream = Some(SendStream(s));
            notify_status(&mut env, state.target_ref.as_obj(), "Listening...");
        }
        Err(e) => {
            notify_status(&mut env, state.target_ref.as_obj(), &format!("Error: {}", e));
        }
    }
}

pub fn stop_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    state.stream = None;

    let buffer = state.audio_buffer.lock().unwrap().clone();
    let jvm = state.jvm.clone();
    let target_ref = state.target_ref.clone();

    notify_status(&mut env, target_ref.as_obj(), "Transcribing...");

    std::thread::spawn(move || {
        let mut env = jvm.attach_current_thread().unwrap();
        let obj = target_ref.as_obj();

        // Wait for model if loading
        if engine::get_engine().is_none() && MODEL_LOADING.load(Ordering::Acquire) {
            notify_status(&mut env, obj, "Waiting for model...");
            let start = std::time::Instant::now();
            while engine::get_engine().is_none() && MODEL_LOADING.load(Ordering::Acquire) {
                if start.elapsed() > std::time::Duration::from_secs(120) {
                    notify_status(&mut env, obj, "Error: timeout waiting for model");
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }

        if let Some(eng_arc) = engine::get_engine() {
            let res = {
                let mut eng = eng_arc.lock().unwrap();
                eng.transcribe_samples(buffer, None)
            };

            match res {
                Ok(r) => {
                    notify_status(&mut env, obj, "Ready");
                    notify_text(&mut env, obj, &r.text);
                }
                Err(e) => notify_status(&mut env, obj, &format!("Error: {}", e)),
            }
        } else {
            notify_status(&mut env, obj, "Error: model failed to load");
        }
    });
}

pub fn cancel_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    state.stream = None;
    state.audio_buffer.lock().unwrap().clear();
    notify_status(&mut env, state.target_ref.as_obj(), "Canceled");
}

