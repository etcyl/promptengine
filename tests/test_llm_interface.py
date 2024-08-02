import unittest
from unittest.mock import patch, MagicMock
from prompt_engine.llm_interface import LLMInterface
from sklearn.datasets import load_iris
from sklearn.svm import SVC

class TestLLMInterface(unittest.TestCase):
    @patch('prompt_engine.local_llm.LocalLLM.__init__', return_value=None)
    def setUp(self, mock_local_llm):
        self.llm_interface = LLMInterface()
        self.llm_interface.handler = MagicMock()
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.model = SVC()

    def test_grid_search(self):
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        best_params, best_score = self.llm_interface.grid_search(self.model, param_grid, self.X, self.y)
        
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_score)
        self.assertIsInstance(best_params, dict)
        self.assertIn('C', best_params)
        self.assertIn('kernel', best_params)
        self.assertIsInstance(best_score, float)

    @patch('openai.ChatCompletion.create')
    def test_generate_text_openai(self, mock_openai_create):
        mock_openai_create.return_value = {'choices': [{'message': {'content': 'Generated text from OpenAI.'}}]}
        llm = LLMInterface(use_openai=True, openai_api_key='test_api_key')
        result = llm.generate_text('Test prompt', max_length=10)
        
        mock_openai_create.assert_called_once_with(
            model="gpt-4",  # Updated to reflect the actual model name used
            messages=[{'role': 'system', 'content': 'Generate text'}, {'role': 'user', 'content': 'Test prompt'}],
            max_tokens=10,
            n=1,
            temperature=0.7
        )

        self.assertEqual(result, ['Generated text from OpenAI.'])

    @patch('prompt_engine.local_llm.LocalLLM', autospec=True)
    def test_generate_text_local(self, MockLocalLLM):
        mock_local_llm_instance = MockLocalLLM.return_value
        mock_local_llm_instance.generate_text.return_value = ["Generated text"]
        llm_interface = LLMInterface(model_name='gpt2', remote=False)
        llm_interface.handler = mock_local_llm_instance
        result = llm_interface.generate_text("Hello, world!")
        mock_local_llm_instance.generate_text.assert_called_once_with("Hello, world!", 50, 1, 0.7, 50, 0.9)
        self.assertEqual(result, ["Generated text"])

    @patch('requests.post')
    @patch('prompt_engine.remote_llm.RemoteLLM')
    def test_generate_text_remote(self, MockRemoteLLM, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = ["Generated text"]
        mock_post.return_value = mock_response
        mock_remote_llm_instance = MockRemoteLLM.return_value
        mock_remote_llm_instance.generate_text.side_effect = lambda prompt, max_length, num_return_sequences, temperature, top_k, top_p: mock_post.return_value.json()
        llm_interface = LLMInterface(remote=True, api_url='http://example.com')
        result = llm_interface.generate_text("Hello, world!")
        mock_post.assert_called_once_with(
            'http://example.com/generate',
            json={
                'prompt': 'Hello, world!',
                'max_length': 50,
                'num_return_sequences': 1,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.9
            }
        )
        self.assertEqual(result, ["Generated text"])

    @patch('optuna.create_study')
    def test_bayesian_optimization(self, MockCreateStudy):
        mock_study_instance = MockCreateStudy.return_value
        mock_study_instance.best_params = {'param': 'value'}
        mock_study_instance.best_value = 0.9
        objective = MagicMock()
        llm_interface = LLMInterface()
        best_params, best_value = llm_interface.bayesian_optimization(objective, n_trials=50)
        MockCreateStudy.assert_called_once_with(direction='maximize')
        mock_study_instance.optimize.assert_called_once_with(objective, n_trials=50)
        self.assertEqual(best_params, {'param': 'value'})
        self.assertEqual(best_value, 0.9)

    def test_train_model(self):
        model = MagicMock()
        X_train = [[1], [2], [3]]
        y_train = [1, 2, 3]
        llm_interface = LLMInterface()
        trained_model = llm_interface.train_model(model, X_train, y_train)
        model.fit.assert_called_once_with(X_train, y_train)
        self.assertEqual(trained_model, model)

    def test_evaluate_model(self):
        model = MagicMock()
        X_test = [[1], [2], [3]]
        y_test = [1, 2, 3]
        model.score.return_value = 0.9
        llm_interface = LLMInterface()
        score = llm_interface.evaluate_model(model, X_test, y_test)
        model.score.assert_called_once_with(X_test, y_test)
        self.assertEqual(score, 0.9)

    @patch('platform.processor')
    def test_detect_amd_hardware(self, MockProcessor):
        MockProcessor.return_value = 'AMD Ryzen'
        llm_interface = LLMInterface()
        result = llm_interface._detect_amd_hardware()
        MockProcessor.assert_called_once()
        self.assertTrue(result)

    @patch('platform.processor')
    def test_detect_intel_hardware(self, MockProcessor):
        MockProcessor.return_value = 'Intel Core'
        llm_interface = LLMInterface()
        result = llm_interface._detect_intel_hardware()
        MockProcessor.assert_called_once()
        self.assertTrue(result)

    @patch('openai.ChatCompletion.create')
    def test_jax_processing(self, mock_openai_create):
        # Mock the API response to return a lowercase string
        mock_openai_create.return_value = {
            'choices': [{'message': {'content': 'jax should uppercase this text.'}}]
        }
        # Assuming use_jax is True and it's initialized properly in LLMInterface
        llm = LLMInterface(use_openai=True, openai_api_key='test_api_key', use_jax=True)
        # Call the method that triggers JAX processing
        result = llm.generate_text('Any prompt', max_length=10)
        
        # Check if the text has been converted to uppercase
        self.assertEqual(result, ['JAX SHOULD UPPERCASE THIS TEXT.'])

if __name__ == '__main__':
    unittest.main()
