def create_dummy_wrapper_filter(filter_class):
    class DummyWrapper(filter_class):
        codec_id = f'dummy_{filter_class.codec_id}'
        
        def decode(self, buf, out=None):
            assert out is not None
            super().decode(super().encode(buf), out)
    
    return DummyWrapper